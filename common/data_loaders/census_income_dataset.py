#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: census_income_dataset.py
@time: 2021/12/21 2:47 PM
@desc: Census Income dataset loader
"""
import os

import numpy as np
import pandas as pd

from common.config import logger
from common.data_loaders.ltr_dataset import LtrDataset

SEED = 1


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


class CensusIncomeDataset(LtrDataset):
    ''' Census Income Dataset '''

    def __init__(self, batch_size, dataset_path, type='train'):
        super(CensusIncomeDataset, self).__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.type = type
        if not os.path.exists(self.dataset_path):
            raise ValueError("Dataset path not exist: {} ".format(self.dataset_path))

        self.train_data, self.train_label, self.validation_data, self.validation_label, self.test_data, self.test_label, self.output_info = self._load_data()
        logger.info(" census income dataset preprocessed ")

    def data_converter(self, inputs):
        return inputs

    def _load_data(self):
        ''' load data and preprocess '''
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

        return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info

    def __getitem__(self, idx):
        if self.type == 'train':
            return self.train_data.iloc[idx], self.train_label.iloc[idx]
        elif self.type == 'val':
            return self.validation_data.iloc[idx], self.validation_label.iloc[idx]
        elif self.type == 'test':
            return self.test_data.iloc[idx], self.test_label.iloc[idx]
        else:
            raise ValueError("Not recognized type {}, must in train, val,test ".format(self.type))

    def __len__(self):
        if self.type == 'train':
            return len(self.train_label)
        elif self.type == 'val':
            return len(self.validation_label)
        elif self.type == 'test':
            return len(self.test_label)
        else:
            raise ValueError("Not recognized type {}, must in train, val,test ".format(self.type))

    # signlon method
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(CensusIncomeDataset, "_instance"):
            CensusIncomeDataset._instance = CensusIncomeDataset(*args, **kwargs)
        return CensusIncomeDataset._instance
