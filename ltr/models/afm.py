#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: anm.py
@time: 2022/1/21 2:19 PM
@desc: Implementation of Attentional Factorization Machines
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.models.layers.interaction import AFMLayer
from ltr.models.layers.core import MLP,EmbeddingLayer

class AFM(nn.Module):
    def __init__(self,
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs):
        super(AFM, self).__init__()

        self.deep_column_idx = deep_column_idx

        self.deep_continuous_cols = deep_continuous_cols

        self.emb = EmbeddingLayer(deep_column_idx=deep_column_idx, deep_continuous_cols=deep_continuous_cols,
                                  deep_emb_inputs=deep_emb_inputs)

        self.linear = nn.Linear(self.emb.out_dim, 1)

        self.pair_interaction = AFMLayer(embedding_size=self.emb.embedding_dim,attention_factor=20,dropout=0.3)

    def forward(self,x):
        deep_inputs = x['deep']
        batch_size = deep_inputs.shape[0]
        cont_outs = deep_inputs[:, [self.deep_column_idx[col] for col in self.deep_continuous_cols]]
        emb_outs = self.emb(deep_inputs).to(torch.float32)
        linear_inputs = torch.cat([cont_outs, emb_outs.view(batch_size, -1)], dim=1).to(torch.float32)
        linear_outs = self.linear(linear_inputs)

        afm_outs = self.pair_interaction(emb_outs)
        linear_outs.add_(afm_outs)
        return linear_outs
