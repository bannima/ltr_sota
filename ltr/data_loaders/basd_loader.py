#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: basd_loader.py
@time: 2021/12/28 5:11 PM
@desc: 
"""

from abc import ABCMeta,abstractmethod

class BaseLoader(metaclass=ABCMeta):
    def __init__(self,batch_size):
        self.batch_size = batch_size

    @property
    @abstractmethod
    def train_loader(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def validation_loader(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def test_loader(self):
        raise NotImplementedError()

    def data_converter(self, inputs):
        ''' do nothing '''
        return inputs

    @property
    def info(self):
        ''' return dataset information where model init needs '''
        return None
