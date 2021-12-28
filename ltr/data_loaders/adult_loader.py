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
from pytorch_widedeep.preprocessing import WidePreprocessor,TabPreprocessor

from ltr.config import logger
from ltr.config import dataset_path
from ltr.data_loaders.basd_loader import BaseLoader

class AdultLoader(BaseLoader):
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.adult_datapath = os.path.join(dataset_path,'adult')
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
        wide_cols = [
            "age_buckets",
            "education",
            "relationship",
            "workclass",
            "occupation",
            "native_country",
            "gender",
        ]
        crossed_cols = [("education", "occupation"), ("native_country", "occupation")]
        # cat_embed_cols = [
        #     ("education", 10),
        #     ("relationship", 8),
        #     ("workclass", 10),
        #     ("occupation", 10),
        #     ("native_country", 10),
        # ]
        cat_embed_cols = [
            "education",
            "relationship",
            "workclass",
            "occupation",
            "native_country",
        ]
        continuous_cols = ["age", "hours_per_week"]
        target = "income_label"
        target = df[target].values
        prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
        x_wide = prepare_wide.fit_transform(df)
        prepare_deep = TabPreprocessor(
            embed_cols=cat_embed_cols, continuous_cols=continuous_cols  # type: ignore[arg-type]
        )
        x_deep = prepare_deep.fit_transform(df)
        logger.info("adult data processed ")

    @property
    def train_loader(self):
        pass

    @property
    def validation_loader(self):
        pass

    @property
    def test_loader(self):
        pass



