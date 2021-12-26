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

from common.data_loaders.census_income_loader import CensusIncomeLoader

__registered_dataloaders = {
    'CensusIncome': {
        'cls': CensusIncomeLoader,
        'intro': 'Census Income Dataset'
    }
}

def create_dataloaders(dataset, batch_size):
    if dataset not in __registered_dataloaders:
        raise ValueError("Not registered dataset {}, must in {}".format(dataset, list(__registered_dataloaders.keys())))
    dataset_loader = __registered_dataloaders[dataset]['cls'](batch_size)

    return (dataset_loader.train_loader,dataset_loader.validation_loader,dataset_loader.test_loader),dataset_loader.data_converter
