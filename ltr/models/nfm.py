#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: nfm.py
@time: 2022/1/20 4:01 PM
@desc: Implementation of NFM: Neural Factorization Machines for Sparse Predictive Analytics
"""
import torch
import torch.nn as nn
from ltr.models.layers.core import MLP,EmbeddingLayer
from ltr.models.layers.interaction import BiInteractionLayer

class NFM(nn.Module):
    def __init__(self,
                 deep_hidden_dims,
                 deep_dropouts,
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs):
        super(NFM, self).__init__()
        self.deep_column_idx = deep_column_idx

        self.deep_continuous_cols = deep_continuous_cols

        self.emb = EmbeddingLayer(deep_column_idx = deep_column_idx, deep_continuous_cols =deep_continuous_cols, deep_emb_inputs = deep_emb_inputs)

        self.linear = nn.Linear(self.emb.out_dim,1)

        self.mlp =  MLP(hidden_dims=[self.emb.embedding_dim] + deep_hidden_dims, dropouts=deep_dropouts)

        self.bi_pooling = BiInteractionLayer()

    def forward(self,x):
        deep_inputs = x['deep']
        batch_size = deep_inputs.shape[0]
        cont_outs = deep_inputs[:,[self.deep_column_idx[col] for col in self.deep_continuous_cols]]
        emb_outs = self.emb(deep_inputs).to(torch.float32)
        linear_inputs = torch.cat([cont_outs,emb_outs.view(batch_size,-1)],dim=1).to(torch.float32)
        linear_outs = self.linear(linear_inputs)

        deep_inputs = self.bi_pooling(emb_outs)
        linear_outs.add_(self.mlp(deep_inputs))
        return linear_outs



