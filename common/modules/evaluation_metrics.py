#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: evaluation_metrics.py
@time: 2021/12/21 1:39 PM
@desc: evaluation metrics for each task
"""

from functools import partial

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

__registered_metrics = {
    'multi_label': {
        'Macro F1': partial(f1_score, average='macro'),
        'Micro F1': partial(f1_score, average='micro'),
        'Weighted F1': partial(f1_score, average='weighted'),
        'Samples F1': partial(f1_score, average='samples')
    },
    'rank': {
        'ndcg': ''
    },
    'classification':{
        'f1':f1_score,
        'precision':precision_score,
        'accuracy':accuracy_score
    }
}


def create_metrics(metrics_type):
    if metrics_type not in __registered_metrics:
        raise ValueError("{} not registered, must in {}".format(metrics_type, list(__registered_metrics.keys())))
    return __registered_metrics[metrics_type]
