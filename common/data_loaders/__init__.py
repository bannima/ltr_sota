#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: __init__.py.py
@time: 2021/12/21 2:47 PM
@desc: 
"""
import os

from torch.utils.data import DataLoader

from common.config import dataset_path
from common.data_loaders.census_income_dataset import CensusIncomeDataset

__registered_dataloaders = {
    'CensusIncome': {
        'cls': CensusIncomeDataset,
        'dir': 'census-income',
        'intro': 'Census Income Dataset'
    }
}


def create_dataloaders(dataset, batch_size):
    if dataset not in __registered_dataloaders:
        raise ValueError("Not registered dataset {}, must in {}".format(dataset, list(__registered_dataloaders.keys())))
    dataset_dir = os.path.join(dataset_path, __registered_dataloaders[dataset]['dir'])
    # dataset = __registered_dataloaders[dataset]['cls'](batch_size, dataset_dir)
    # signlon method
    train_dataset = __registered_dataloaders[dataset]['cls'].instance(batch_size, dataset_dir,type='train')
    val_dataset = __registered_dataloaders[dataset]['cls'].instance(batch_size, dataset_dir,type='val')
    test_dataset = __registered_dataloaders[dataset]['cls'].instance(batch_size, dataset_dir,type='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )
    return (train_loader, val_loader, test_loader), train_dataset.data_converter
