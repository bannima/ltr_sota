#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: adult_nfm_exp.py
@time: 2022/1/20 5:50 PM
@desc: 
"""

import os
from ltr.config import logger
from ltr.modules.utils import parse_parmas
from ltr.data_loaders import create_dataloaders
from ltr.models import NFM
from ltr.trainers.base_trainer import Trainer
from ltr.modules.analyzer import SingleTaskExpAnalyzer


def train_nfm_with_adult(HYPERS):
    # 1. load adult data
    # wide model cross product features
    crossed_cols = [("education", "occupation"), ("native_country", "occupation")]
    # wide model used featuers
    wide_cols = [
        "age_buckets",
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
        "gender",
    ]

    # deep model features
    continuous_cols = ["age", "hours_per_week"]
    # deep model category embedding features
    cat_embed_cols = [
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
    ]
    (train_loader, val_loader, test_loader), data_converter, info = create_dataloaders(dataset='Adult_DeepCross',
                                                                                       batch_size=HYPERS['Batch'],
                                                                                       dense_cols=continuous_cols,
                                                                                       cat_embed_cols=cat_embed_cols)

    # 2. prepare NFM model
    deepfm = NFM(
        deep_hidden_dims=[1024, 512, 256, 1],
        deep_dropouts=[0.3, 0.3, 0.3, 0.3],
        deep_column_idx=info['deep_column_idx'],
        deep_continuous_cols = info['deep_continuous_cols'],
        deep_emb_inputs =info['deep_emb_inputs'],
    )
    logger.info(" NFM initialized ")

    # 3. train mode use Single Task Trainer
    result_path = os.path.join(os.path.dirname(__file__), 'results')
    trainer = Trainer(
        model=deepfm,
        dataloaders=(train_loader, val_loader, test_loader),
        data_converter=data_converter,
        result_path=result_path,
        HYPERS=HYPERS,
    )
    epoch_stats_file = trainer.fit()

    # 4. trained nfm model analysis using ExperimentAnalyzer
    cur_dir = os.path.dirname(__file__)
    analyzer = SingleTaskExpAnalyzer(os.path.join(cur_dir, epoch_stats_file))
    analyzer.analysis_experiment(exp_result_dir=trainer.exp_result_dir, title='NFM_Adult_Experiment')

if __name__ == '__main__':
    logger.info(" Start train Neural Factorization Machines on Adult data ")
    HYPERS = parse_parmas()

    HYPERS['Epochs'] = 10
    HYPERS['LearningRate'] = 2e-3
    HYPERS['Batch'] = 256
    HYPERS['Save_Model'] = False
    HYPERS['Criterion']='BCEWithLogitsLoss'

    train_nfm_with_adult(HYPERS)


