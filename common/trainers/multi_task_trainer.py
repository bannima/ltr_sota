#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: multi_task_trainer.py
@time: 2021/12/22 10:23 AM
@desc: 
"""
import numpy as np
from common.trainers.base_trainer import BaseTrainer
from common.config import logger

class MultiTaskTrainer(BaseTrainer):
    '''MMoE trainers'''
    def __init__(self,
                 model,
                 dataloaders,
                 data_converter,
                 result_path,
                 HYPERS,
                 num_tasks
                 ):
        super(MultiTaskTrainer, self).__init__(
                 model,
                 dataloaders,
                 data_converter,
                 result_path,
                 HYPERS)
        self.num_tasks = num_tasks

    def calc_loss(self, outputs, labels):
        ''' multi task loss calculation '''
        y1, y2 = labels[:, 0], labels[:, 1]
        y_1, y_2 = outputs[0], outputs[1]
        loss1 = self.criterion(y_1, y1)
        loss2 = self.criterion(y_2, y2)
        batch_loss = loss1 + loss2
        return batch_loss

    def calc_predicts(self, outputs, labels):
        ''' multi task predict labels '''
        # move logits and labels to GPU
        logits = [output.detach().cpu().numpy() for output in outputs]
        labels = labels.to("cpu")
        y_pred = [list(np.argmax(logit, axis=1).flatten()) for logit in logits]
        label_ids = [list(np.argmax(labels[:,i,:],axis=1).numpy()) for i in range(self.num_tasks)]
        return y_pred,label_ids

    def calc_metrics(self,predict_label,target_label,metrics):
        ''' calc metrics for multi task situation '''
        eval_metrics = {}
        for task in range(self.num_tasks):
            eval_metrics[task] = {}
            #task predicts
            task_preds = predict_label[task]
            #task labels
            task_labels = target_label[task]

            for metric_name, metric in metrics.items():
                try:
                    eval_metrics[task][metric_name] = metric(task_preds, task_labels)
                except Exception as e:
                    pass

        return eval_metrics

    def transform_result(self, result):
        result = np.hstack(result)
        return result.tolist()

    def report_metrics(self,metrics_results):
        ''' report the eval and test metrics '''
        for task in metrics_results:
            logger.info("# Task {} metrics".format(task))
            for metric in metrics_results[task]:
                logger.info("{}: {}".format(metric,metrics_results[task][metric]))
