#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: census_income_mmoe_trainer.py
@time: 2021/12/20 4:42 PM
@desc: 
"""
import os

from common.config import logger
from common.data_loaders import create_dataloaders
from common.utils import parse_parmas
from ltr_sota import MMoE
from common.trainers.multi_task_trainer import MultiTaskTrainer

def train_mmoe_with_censusincome(HYPERS):
    # 1. load census income dataset
    (train_loader, val_loader, test_loader), data_converter = create_dataloaders(dataset='CensusIncome',
                                                                                 batch_size=HYPERS['Batch'])
    logger.info(" census income dataset loaded")

    # 2. prepare MMoE model
    mmoe = MMoE(
        input_size=499,
        num_experts=6,
        experts_out=16,
        experts_hidden=32,
        towers_hidden=8,
        output_size=2,
        tasks=2
    )
    logger.info(" mmoe initialized ")

    # 3. train mode use MMoE Trainer
    trainer = MultiTaskTrainer(
        model=mmoe,
        dataloaders=(train_loader, val_loader, test_loader),
        data_converter=data_converter,
        result_path=os.path.join(os.path.dirname(__file__), 'results'),
        HYPERS=HYPERS,
        num_tasks=2 #当前数据集只有两个任务
    )
    trainer.run_epoch()

    # 4. eval results on census income dataset

    # 5. trained mmoe model analysis


if __name__ == '__main__':
    logger.info(" Start train MMoE on census income dataset ")
    HYPERS = parse_parmas()

    HYPERS['Epochs'] = 100
    HYPERS['Save_Model']=False

    train_mmoe_with_censusincome(HYPERS)
