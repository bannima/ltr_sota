#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: adult_afm_exp.py
@time: 2022/1/21 3:25 PM
@desc: 
"""
import os
from ltr.config import logger
from ltr.modules.utils import parse_parmas
from ltr.data_loaders import create_dataloaders
from ltr.models import AFM
from ltr.trainers.base_trainer import Trainer
from ltr.modules.analyzer import SingleTaskExpAnalyzer

def train_afm_with_adult(HYPERS):
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

    # 2. prepare AFM model
    afm = AFM(
        deep_column_idx=info['deep_column_idx'],
        deep_continuous_cols = info['deep_continuous_cols'],
        deep_emb_inputs =info['deep_emb_inputs'],
    )
    logger.info(" AFM initialized ")

    # 3. train mode use Single Task Trainer
    result_path = os.path.join(os.path.dirname(__file__), 'results')
    trainer = Trainer(
        model=afm,
        dataloaders=(train_loader, val_loader, test_loader),
        data_converter=data_converter,
        result_path=result_path,
        HYPERS=HYPERS,
    )
    epoch_stats_file = trainer.fit()

    # 4. trained afm model analysis using ExperimentAnalyzer
    cur_dir = os.path.dirname(__file__)
    analyzer = SingleTaskExpAnalyzer(os.path.join(cur_dir, epoch_stats_file))
    analyzer.analysis_experiment(exp_result_dir=trainer.exp_result_dir, title='AFM_Adult_Experiment')

if __name__ == '__main__':
    logger.info(" Start train Attentional Factorization Machines on Adult data ")
    HYPERS = parse_parmas()

    HYPERS['Epochs'] = 12
    HYPERS['LearningRate'] = 5e-4
    HYPERS['Batch'] = 256
    HYPERS['Save_Model'] = False
    HYPERS['Criterion']='BCEWithLogitsLoss'

    train_afm_with_adult(HYPERS)


