#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: adult_widedeep_exp.py.py
@time: 2021/12/27 7:55 PM
@desc: 
"""
import os
from ltr.config import logger
from ltr.modules.utils import parse_parmas
from ltr.data_loaders import create_dataloaders
from ltr.models import WideDeep
from ltr.trainers.base_trainer import Trainer
from ltr.modules.analyzer import ExperimentAnalyzer


def train_widedeep_with_adult(HYPERS):
    # 1. load adult dataset
    (train_loader, val_loader, test_loader), data_converter = create_dataloaders(dataset='Adult',
                                                                                 batch_size=HYPERS['Batch'])
    logger.info(" Adult dataset loaded")

    # 2. prepare MMoE model
    widedeep = WideDeep(
        wide_dim=805,
        pred_dim=1,
        deep_hidden_dims=[805, 1024, 512, 256, 1],
        deep_dropouts=[0.3, 0.3, 0.3, 0.3, 0.3],
    )
    logger.info(" Wide&Deep initialized ")

    # 3. train mode use Single Task Trainer
    result_path = os.path.join(os.path.dirname(__file__), 'results')
    trainer = Trainer(
        model=widedeep,
        dataloaders=(train_loader, val_loader, test_loader),
        data_converter=data_converter,
        result_path=result_path,
        HYPERS=HYPERS,
    )
    epoch_stats_file = trainer.fit()
    # epoch_stats_file = os.path.join(project_path,'examples/census_income_mmoe/results/Model_LR1e-05_Batch64_LossBCELoss/Epoch_Statstics_Time20211226_1657.csv')

    # 4. trained mmoe model analysis using MultiTaskExpAnalyzer
    cur_dir = os.path.dirname(__file__)
    analyzer = ExperimentAnalyzer(os.path.join(cur_dir, epoch_stats_file))
    analyzer.analysis_experiment(exp_result_dir=trainer.exp_result_dir, title='WideDeep_Adult_Experiment')


if __name__ == '__main__':
    logger.info(" Start train Wide&Deep on Adult dataset ")
    HYPERS = parse_parmas()

    HYPERS['Epochs'] = 2
    HYPERS['LearningRate'] = 1e-3
    HYPERS['Batch'] = 256
    HYPERS['Save_Model'] = False

    train_widedeep_with_adult(HYPERS)
