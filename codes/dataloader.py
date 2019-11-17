#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import Sampler

import math


class BucketBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        # >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        # >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class TrainDataset(Dataset):

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, KB = None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.KB = KB
        self.max_nbrs = 500
        self.use_neighbors = True
        if KB == None:
            self.use_neighbors = False

        self.triples = sorted(self.triples, key = lambda x: -len(self.KB.e1_view[x[0]]) )

        self.__getitem__(5)



        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]


        head, relation, tail = positive_sample

        if self.use_neighbors:
            pos_neighbors = self.KB.e1_view[head]
            neighboring_relations = []
            neighboring_entities = []
            count = 0
            for r,e2 in pos_neighbors:
                neighboring_entities.append(e2)
                neighboring_relations.append(r)
                count += 1
                if count > self.max_nbrs:
                    break



        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]


        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
        if self.use_neighbors:
            return positive_sample, negative_sample, subsampling_weight, self.mode, neighboring_entities, neighboring_relations
        else:
            return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        if len(data) > 4:
            all_neighboring_entities = [_[4] for _ in data]
            all_neighboring_lens = [len(_[4]) for _ in data]
            all_neighboring_relations = [_[5] for _ in data]
            neighboring_r = np.zeros((len(all_neighboring_entities), max(all_neighboring_lens)+1)) #B, max_nbrs
            neighboring_r_mask = np.ones((len(all_neighboring_entities), max(all_neighboring_lens)+1)) #B, max_nbrs
            neighboring_e = np.zeros((len(all_neighboring_entities), max(all_neighboring_lens)+1)) #B, max_nbrs
            neighboring_e_mask = np.ones((len(all_neighboring_entities), max(all_neighboring_lens)+1))  # B, max_nbrs
            for b in range(len(all_neighboring_entities)):

                for nbr_count, nbr_e in enumerate(all_neighboring_entities[b]):
                    neighboring_e[b, nbr_count] = nbr_e
                    neighboring_e_mask[b, nbr_count] = 0

                    neighboring_r[b, nbr_count] = all_neighboring_relations[b][nbr_count]
                    neighboring_r_mask[b, nbr_count] =  0

            neighboring_e = torch.LongTensor(neighboring_e)
            neighboring_e_mask = torch.ByteTensor(neighboring_e_mask)
            neighboring_r = torch.LongTensor(neighboring_r)
            neighboring_r_mask = torch.ByteTensor(neighboring_r_mask)
            return positive_sample, negative_sample, subsample_weight, mode, neighboring_e, neighboring_r, neighboring_e_mask, neighboring_r_mask
        else:
            return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail





class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode, KB=None):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.max_nbrs = 500
        self.use_neighbors = True
        self.triples = sorted(self.triples, key = lambda x: -len(KB.e1_view[x[0]]) )

        if KB == None:
            self.use_neighbors = False
        else:
            self.KB = KB

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        neighboring_relations = []
        neighboring_entities = []
        if self.use_neighbors:
            pos_neighbors = self.KB.e1_view[head]

            count = 0
            for r,e2 in pos_neighbors:
                neighboring_entities.append(e2)
                neighboring_relations.append(r)
                count += 1
                if count >= self.max_nbrs:
                    break

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        if self.use_neighbors:
            return positive_sample, negative_sample, filter_bias, self.mode, neighboring_entities, neighboring_relations
        else:
            return positive_sample, negative_sample, filter_bias, self.mode

    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        if len(data[0]) > 4:
            all_neighboring_entities = [_[4] for _ in data]
            all_neighboring_lens = [len(_[4]) for _ in data]
            all_neighboring_relations = [_[5] for _ in data]
            neighboring_r = np.zeros((len(all_neighboring_entities), max(all_neighboring_lens)+1))  # B, max_nbrs
            neighboring_r_mask = np.ones((len(all_neighboring_entities), max(all_neighboring_lens)+1))  # B, max_nbrs
            neighboring_e = np.zeros((len(all_neighboring_entities), max(all_neighboring_lens)+1))  # B, max_nbrs
            neighboring_e_mask = np.ones((len(all_neighboring_entities), max(all_neighboring_lens)+1))  # B, max_nbrs
            for b in range(len(all_neighboring_entities)):

                for nbr_count, nbr_e in enumerate(all_neighboring_entities[b]):
                    neighboring_e[b, nbr_count] = nbr_e
                    neighboring_e_mask[b, nbr_count] = 0

                    neighboring_r[b, nbr_count] = all_neighboring_relations[b][nbr_count]
                    neighboring_r_mask[b, nbr_count] = 0

            neighboring_e = torch.LongTensor(neighboring_e)
            neighboring_e_mask = torch.ByteTensor(neighboring_e_mask)
            neighboring_r = torch.LongTensor(neighboring_r)
            neighboring_r_mask = torch.ByteTensor(neighboring_r_mask)
            return positive_sample, negative_sample, filter_bias, mode, neighboring_e, neighboring_r, neighboring_e_mask, neighboring_r_mask
        else:

            return positive_sample, negative_sample, filter_bias, mode


class TestDataset_MINERVA(Dataset):
    def __init__(self, triples, candidate_entities, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        from collections import defaultdict
        self.candidate_entities = candidate_entities

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set and rand_tail in self.candidate_entities[(head, relation)]
                   else (-1, tail) for rand_tail in range(self.nentity)]
            if tail in self.candidate_entities[(head, relation)]:
                tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode
    def debug(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            import pdb
            pdb.set_trace()
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set and rand_tail in self.candidate_entities[(head, relation)]
                   else (-1, tail) for rand_tail in range(self.nentity)]
            if tail in self.candidate_entities[(head, relation)]:
                tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class OneShotIterator(object):
    def __init__(self, dataloader_tail):
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data