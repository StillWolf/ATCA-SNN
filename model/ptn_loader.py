"""
-*- coding: utf-8 -*-

@Time    : 2021-04-13 10:47

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : ptn_loader.py
"""

import torch
import numpy as np
import torch.nn as nn
import os
import soundfile
from torch.utils.data import DataLoader, Dataset


class sPtn(Dataset):
    def __init__(self, dataSource, labelSource, TmaxSource, mode='spike', dt=None):
        self.mode = mode
        if mode == 'spike':
            """
            dataSource: num_samples x max_num_spikes x 2, pad with -1 for inconsistency in spiking counts.
                        eg:
                            num_samples = 2 
                            max_num_spikes = 5
                            sample 1:
                                for neuron 1 , the afferent is [2, 5, 7]
                                for neuron 2 , the afferent is [1, 9]
                            sample 2: 
                                for neuron 1 : the afferent is [2, 9, 14]
                                for neuron 2 , the afferent is [6]
                            ==============================================================================
                            then dataSource = [
                                                [[1,2],[1,5],[1,7],[2,1],[2,9]], # for sample 1 
                                                [[1,2],[1,9],[1,14],[2,6],[-1,-1]] # for sample 2
                                                                ]]
            """
            self.dataSource = dataSource
            self.labelSource = labelSource
            self.TmaxSource = TmaxSource
        elif mode == 'neuron':
            assert not dt is None, 'for neuron-based format, dt should not be None for iterative SNNs'
            self.dt = dt
            self.nb_steps = int(np.ceil(TmaxSource.max() / dt)) + 1
            self.dataSource = dataSource
            self.labelSource = labelSource
            self.TmaxSource = TmaxSource
        elif mode == 'time':
            raise NotImplementedError('Time-based loading is not supported util now.')
        else:
            raise NotImplementedError("Only 'spike' ,'neuron' and 'time' format are in the plan.")

    def __getitem__(self, index):
        if self.mode == 'spike':
            spike = self.dataSource[index, :, :]
            spike = np.transpose(spike)
            label = self.labelSource[index]
            Tmax = float(self.TmaxSource[index])
            return spike, Tmax, label
        elif self.mode == 'neuron':
            data = self.dataSource[index]
            ptn = np.zeros([len(data), self.nb_steps])
            for i, aff in enumerate(data):
                aff = np.array(aff).reshape([-1])
                # print(aff)
                if len(aff) > 0:
                    ptn[np.ones_like(aff, dtype=np.int) * i, np.round(aff / self.dt).astype(np.int)] = 1
            Tmax = float(self.TmaxSource[index])
            label = int(self.labelSource[index]) - 1
            return ptn, label

    def __len__(self):
        return self.dataSource.shape[0]


class sPtn_TCA(Dataset):
    def __init__(self, dataSource, labelSource, TmaxSource, TCA_slice, dt=None):
        self.dt = dt
        self.nb_steps = int(np.ceil(TmaxSource.max() / dt)) + 1
        self.dataSource = dataSource
        self.labelSource = labelSource
        self.TmaxSource = TmaxSource
        self.data_delta = int(self.nb_steps / 10)
        self.TCA_slice = TCA_slice

    def __getitem__(self, index):
        ptn = np.zeros([len(self.dataSource[0]), self.nb_steps * self.TCA_slice + (self.TCA_slice-1) * self.data_delta])
        labels = []
        for t in range(self.TCA_slice):
            index_tca = np.random.randint(self.dataSource.shape[0])
            while self.labelSource[index_tca] - 1 in labels:
                index_tca = np.random.randint(self.dataSource.shape[0])
            data = self.dataSource[index_tca]
            for i, aff in enumerate(data):
                aff = np.array(aff).reshape([-1])
                if len(aff) > 0:
                    ptn[np.ones_like(aff, dtype=np.int) * i, np.round(aff / self.dt).astype(np.int)+t*(self.nb_steps+self.data_delta)] = 1
            labels.append(int(self.labelSource[index_tca]) - 1)
        return ptn, labels

    def __len__(self):
        return int(self.dataSource.shape[0] / self.TCA_slice)