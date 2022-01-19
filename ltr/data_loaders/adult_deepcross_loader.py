#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: adult_deepcross_loader.py
@time: 2022/1/2 9:46 PM
@desc: 
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pytorch_widedeep.preprocessing import WidePreprocessor,TabPreprocessor

from ltr.config import logger
from ltr.config import dataset_path
from ltr.data_loaders.basd_loader import BaseLoader

def wd_train_test_split(x_deep,target,test_size=0.3,seed=2022):
    ''' split the deep dataset and return train,val,test set '''
    y_train,y_eval_test,train_idx,test_eval_idx = train_test_split(
        target,
        np.arange(len(target)),
        test_size = test_size,
        random_state=seed
    )
    y_test,y_eval,test_idx,eval_idx = train_test_split(
        y_eval_test,
        np.arange(len(y_eval_test)),
        test_size=test_size,
        random_state=seed
    )
    train_set = DeepCrossDataset(
        deep = x_deep[train_idx],
        target = target[train_idx]
    )
    val_set = DeepCrossDataset(
        deep = x_deep[eval_idx],
        target = target[eval_idx]
    )
    test_set = DeepCrossDataset(
        deep=x_deep[test_idx],
        target = target[test_idx]
    )
    return train_set,val_set,test_set

class DeepCrossDataset(Dataset):
    def __init__(self,deep,target):
        super(DeepCrossDataset, self).__init__()
        self.deep = deep
        self.target = target

    def __getitem__(self, idx):
        x = Bunch()
        x.deep = self.deep[idx]
        if self.target is not None:
            return x,self.target[idx]
        else:
            return x

    def __len__(self):
        if self.deep is not None:
            return len(self.deep)
        elif self.target is not None:
            return len(self.target)
        else:
            raise ValueError("dataset length not recognized")

class AdultDeepCrossLoader(BaseLoader):
    def __init__(self,
                 batch_size,
                 dense_cols, #dense feature columns
                 cat_embed_cols, # category embedding columns
                 cont_feat_norm = True # normalize continuous features
                 ):
        super(AdultDeepCrossLoader, self).__init__(batch_size)
        self.adult_datapath = os.path.join(dataset_path,'adult')

        #deep model embedding features
        self.cat_embed_cols = cat_embed_cols
        #dense feature vectors
        self.dense_cols = dense_cols

        self.cont_feat_norm = cont_feat_norm

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        self.preprocess()

    def preprocess(self):
        df = pd.read_csv(os.path.join(self.adult_datapath,'adult.csv.zip'))
        df.columns = [c.replace("-", "_") for c in df.columns]
        df["age_buckets"] = pd.cut(
            df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
        )
        df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
        df.drop("income", axis=1, inplace=True)
        df.head()

        #continuous feature normalization
        if self.cont_feat_norm:
            mms = MinMaxScaler(feature_range=(0,1))
            df[self.dense_cols] = mms.fit_transform(df[self.dense_cols])

        target = "income_label"
        target = df[target].values

        self.prepare_deep = TabPreprocessor(
            embed_cols=self.cat_embed_cols, continuous_cols=self.dense_cols, auto_embed_dim=False,default_embed_dim=32 # type: ignore[arg-type]
        )
        x_deep = self.prepare_deep.fit_transform(df)

        #counstruct train,val,test
        self.train,self.val,self.test = wd_train_test_split(x_deep,target)

        logger.info("adult data processed ")

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._train_loader =  DataLoader(
                dataset=self.train,
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._train_loader

    @property
    def validation_loader(self):
        if self._val_loader is None:
            self._val_loader =  DataLoader(
                dataset=self.val,
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._val_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                dataset=self.test,
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._test_loader

    @property
    def info(self):
        ''' return adult dataset information where model init needs'''
        embed_dims = sum([col[2] for col in self.prepare_deep.embeddings_input])+len(self.prepare_deep.continuous_cols)
        info = {
            'deep_dims':embed_dims,
            'deep_column_idx':self.prepare_deep.column_idx, #total deep model features and idx
            'deep_continuous_cols':self.prepare_deep.continuous_cols,# deep continuous features
            'deep_emb_inputs':self.prepare_deep.embeddings_input # deep embedding layers and dims
        }
        return info


