#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: ltr_dataset.py
@time: 2021/12/21 7:06 PM
@desc: 
"""
from abc import abstractmethod

from torch.utils.data import Dataset


class LtrDataset(Dataset):
    ''' Ltr Dataset must implements this abstract class '''

    def __init__(self):
        super(LtrDataset, self).__init__()

    @abstractmethod
    def data_converter(self, inputs):
        '''runtime data converter, before train in GPU'''
        raise NotImplementedError()
