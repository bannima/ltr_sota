#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: ltr_trainer.py
@time: 2022/1/2 7:28 PM
@desc: 
"""
import torch
import numpy as np
import pandas as pd
from ltr.config import logger
from ltr.trainers.base_trainer import Trainer

class LtrTrainer(Trainer):
    '''Learning to rank trainers'''
    def __init__(self,
                 model,
                 dataloaders,
                 data_converter,
                 result_path,
                 HYPERS,
                 ):
        super(LtrTrainer, self).__init__(
            model,
            dataloaders,
            data_converter,
            result_path,
            HYPERS)

    def calc_loss(self, outputs, labels):
        '''single task loss calculation, can be override'''
        labels = labels.view(-1,1) #tranform [batch_size] to [batch_size X 1]
        labels = labels.to(torch.float32)
        batch_loss = self.criterion(outputs, labels)
        return batch_loss

    def calc_predicts(self, outputs, labels):
        ''' single task prediction calculation, can be override'''
        # move logits and labels to GPU
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.to("cpu")
        y_pred = logits.squeeze()
        return y_pred, label_ids

    def calc_metrics(self, predict_label, target_label, metrics,group_ids=None):
        ''' calc  metrics for ltr task situation '''

        if group_ids is not None:
            group_ids = np.array(group_ids)

        df = pd.DataFrame(np.stack((group_ids, predict_label, target_label), axis=1),
                          columns=('group_id', 'predict_label', 'target_label'))
        eval_metrics = {}
        for group_id in np.unique(df['group_id']):
            group = df[df['group_id']==group_id]
            for metric_name,metric in metrics.items():
                if metric_name not in eval_metrics:
                    eval_metrics[metric_name] = []
                #note must in (y_true,y_predict) order for ndcg
                eval_metrics[metric_name].append(metric(group['predict_label'],group['target_label']))

        return {key:np.mean(val) for key,val in eval_metrics.items()}

    def report_metrics(self, metrics_results):
        ''' report the eval and test metrics '''
        for task in metrics_results:
            logger.info("# Task {} metrics".format(task))
            for metric in metrics_results[task]:
                logger.info("{}: {}".format(metric, metrics_results[task][metric]))
