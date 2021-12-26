#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: census_income_loader.py
@time: 2021/12/26 3:22 PM
@desc: 
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset

from common.config import logger
from common.config import project_path

SEED = 1

# def getTensorDataset(my_x, my_y):
#     tensor_x = torch.Tensor(my_x)
#     tensor_y = torch.Tensor(my_y)
#     return torch.utils.data.TensorDataset(tensor_x, tensor_y)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()  # 拉成一维矩阵
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class CensusIncomeLoader():
    ''' Census Income dataset loader generator '''
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.dataset_path = os.path.join(project_path,'dataset/census-income')
        self.type = type
        if not os.path.exists(self.dataset_path):
            raise ValueError("Dataset path not exist: {} ".format(self.dataset_path))

        self.train_data, self.train_label, self.validation_data, self.validation_label, self.test_data, self.test_label, self.output_info = self.do_preparation()
        self._train_loader,self._validation_loader,self._test_loader = None,None,None
        logger.info(" census income dataset preprocessed ")

    def do_preparation(self):
        logger.info("### Warning prepeocess census income dataset")
        train_file = os.path.join(self.dataset_path, 'census-income.data.gz')
        test_file = os.path.join(self.dataset_path, 'census-income.test.gz')
        column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour',
                        'hs_college',
                        'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex',
                        'union_member',
                        'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                        'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                        'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                        'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                        'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

        # Load the dataset in Pandas
        train_df = pd.read_csv(
            train_file,
            delimiter=',',
            header=None,
            index_col=None,
            names=column_names
        )
        test_df = pd.read_csv(
            test_file,
            delimiter=',',
            header=None,
            index_col=None,
            names=column_names
        )
        # preprocess dataset
        # First group of tasks according to the paper
        label_columns = ['income_50k', 'marital_stat']

        # One-hot encoding categorical columns
        categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college',
                               'major_ind_code',
                               'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                               'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
                               'det_hh_fam_stat',
                               'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
                               'mig_prev_sunbelt',
                               'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                               'vet_question']
        train_raw_labels = train_df[label_columns]
        other_raw_labels = test_df[label_columns]
        transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
        transformed_other = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)

        # One-hot encoding categorical labels
        train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
        train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
        other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
        other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

        # Filling the missing column in the other set
        transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

        dict_outputs = {
            'income': train_income.shape[1],
            'marital': train_marital.shape[1]
        }
        dict_train_labels = {
            'income': train_income,
            'marital': train_marital
        }
        dict_other_labels = {
            'income': other_income,
            'marital': other_marital
        }
        output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

        # Split the other dataset into 1:1 validation to test according to the paper
        validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
        test_indices = list(set(transformed_other.index) - set(validation_indices))
        validation_data = transformed_other.iloc[validation_indices]
        validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
        test_data = transformed_other.iloc[test_indices]
        test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
        train_data = transformed_train
        train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

        # column stack
        # train_label = np.column_stack((np.argmax(train_label[0], axis=1), np.argmax(train_label[1], axis=1)))
        # validation_label = np.column_stack((np.argmax(validation_label[0],axis=1),np.argmax(validation_label[1],axis=1)))
        # test_label = np.column_stack((np.argmax(test_label[0],axis=1),np.argmax(test_label[1],axis=1)))

        return torch.Tensor(train_data.to_numpy()), torch.Tensor(train_label).permute([1,0,2]), \
               torch.Tensor(validation_data.to_numpy()),  torch.Tensor(validation_label).permute([1,0,2]), \
               torch.Tensor(test_data.to_numpy()), torch.Tensor(test_label).permute([1,0,2]),\
               output_info

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._train_loader =  DataLoader(
                dataset=TensorDataset(self.train_data,self.train_label),
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._train_loader

    @property
    def validation_loader(self):
        if self._validation_loader is None:
            self._validation_loader = DataLoader(
                dataset=TensorDataset(self.validation_data, self.validation_label),
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._validation_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                dataset=TensorDataset(self.validation_data, self.validation_label),
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._test_loader

    def data_converter(self, inputs):
        ''' do nothing for census income dataset '''
        return inputs
