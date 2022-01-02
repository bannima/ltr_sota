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
from ltr.modules.analyzer import SingleTaskExpAnalyzer
from sklearn.metrics import ndcg_score


def train_widedeep_with_adult(HYPERS):
    # 1. load adult data

    #wide model cross product features
    crossed_cols = [("education", "occupation"),("native_country", "occupation")]
    #wide model used featuers
    wide_cols = [
        "age_buckets",
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
        "gender",
    ]

    #deep model features
    continuous_cols = ["age", "hours_per_week"]
    #deep model category embedding features
    cat_embed_cols = [
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
    ]
    (train_loader, val_loader, test_loader), data_converter,info = create_dataloaders(dataset='Adult',
                                                                                 batch_size=HYPERS['Batch'],
                                                                                 wide_cols = wide_cols,
                                                                                 crossed_cols=crossed_cols,
                                                                                 continuous_cols = continuous_cols,
                                                                                 cat_embed_cols= cat_embed_cols)
    logger.info(" Adult data loaded")

    # 2. prepare Wide&Deep model
    widedeep = WideDeep(
        pred_dim=1,
        deep_hidden_dims=[1024, 512, 256,1],
        deep_dropouts=[0.3, 0.3, 0.3, 0.3, 0.3],
        **info
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

    # 4. trained wide deep model analysis using ExperimentAnalyzer
    cur_dir = os.path.dirname(__file__)
    analyzer = SingleTaskExpAnalyzer(os.path.join(cur_dir, epoch_stats_file))
    analyzer.analysis_experiment(exp_result_dir=trainer.exp_result_dir, title='WideDeep_Adult_Experiment')

if __name__ == '__main__':
    logger.info(" Start train Wide&Deep on Adult data ")
    HYPERS = parse_parmas()

    HYPERS['Epochs'] = 100
    HYPERS['LearningRate'] = 1e-3
    HYPERS['Batch'] = 256
    HYPERS['Save_Model'] = False
    HYPERS['Criterion']='BCEWithLogitsLoss'

    train_widedeep_with_adult(HYPERS)
