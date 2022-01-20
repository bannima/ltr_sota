#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: adult_loader.py
@time: 2021/12/27 8:06 PM
@desc: Adult Data Set Loader Generator
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pytorch_widedeep.preprocessing import WidePreprocessor,TabPreprocessor

from ltr.config import logger
from ltr.config import dataset_path
from ltr.data_loaders.base_loader import BaseLoader

def wd_train_test_split(x_wide,x_deep,target,test_size=0.3,seed=2022):
    ''' split the wide deep dataset and return train,val,test set '''
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
    train_set = WideDeepDataset(
        wide = x_wide[train_idx],
        deep = x_deep[train_idx],
        target = target[train_idx]
    )
    val_set = WideDeepDataset(
        wide = x_wide[eval_idx],
        deep = x_deep[eval_idx],
        target = target[eval_idx]
    )
    test_set = WideDeepDataset(
        wide=x_wide[test_idx],
        deep=x_deep[test_idx],
        target = target[test_idx]
    )
    return train_set,val_set,test_set

class WideDeepDataset(Dataset):
    def __init__(self,wide,deep,target):
        super(WideDeepDataset, self).__init__()
        self.wide = wide
        self.deep = deep
        self.target = target

    def __getitem__(self, idx):
        x = Bunch()
        x.wide = self.wide[idx]
        x.deep = self.deep[idx]
        if self.target is not None:
            return x,self.target[idx]
        else:
            return x

    def __len__(self):
        if self.wide is not None:
            return len(self.wide)
        elif self.deep is not None:
            return len(self.deep)
        elif self.target is not None:
            return len(self.target)
        else:
            raise ValueError("dataset length not recognized")

class AdultLoader(BaseLoader):
    def __init__(self,batch_size,
                  wide_cols,
                  crossed_cols ,
                  continuous_cols,
                  cat_embed_cols
                 ):
        super(AdultLoader,self).__init__(batch_size)
        self.adult_datapath = os.path.join(dataset_path,'adult')

        #wide model features
        self.wide_cols = wide_cols
        #wide model cross prodcut features
        self.crossed_cols = crossed_cols

        #deep model features
        self.continuous_cols = continuous_cols
        #deep model embedding features
        self.cat_embed_cols = cat_embed_cols

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

        target = "income_label"
        target = df[target].values
        self.prepare_wide = WidePreprocessor(wide_cols=self.wide_cols, crossed_cols=self.crossed_cols)
        x_wide = self.prepare_wide.fit_transform(df)
        self.prepare_deep = TabPreprocessor(
            embed_cols=self.cat_embed_cols, continuous_cols=self.continuous_cols  # type: ignore[arg-type]
        )
        x_deep = self.prepare_deep.fit_transform(df)

        #counstruct train,val,test
        self.train,self.val,self.test = wd_train_test_split(x_wide,x_deep,target)

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
        info = {
            'wide_dim':self.prepare_wide.wide_dim, # wide input embedding dims
            'deep_column_idx':self.prepare_deep.column_idx, #total deep model features and idx
            'deep_continuous_cols':self.prepare_deep.continuous_cols,# deep continuous features
            'deep_emb_inputs':self.prepare_deep.embeddings_input # deep embedding layers and dims
        }
        return info


