#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: ifm.py
@time: 2022/1/25 6:13 PM
@desc: Implementation of Input-aware Factorization Machine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.models.layers.core import EmbeddingLayer,MLP
from ltr.models.layers.interaction import FMLayer

class IFM(nn.Module):
    def __init__(self,
                 deep_hidden_dims,
                 deep_dropouts,
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs):
        super(IFM, self).__init__()

        self.deep_column_idx = deep_column_idx

        self.deep_continuous_cols = deep_continuous_cols

        self.emb = EmbeddingLayer(deep_column_idx=deep_column_idx, deep_continuous_cols=deep_continuous_cols,
                                  deep_emb_inputs=deep_emb_inputs)

        self.linear = nn.Linear(self.emb.out_dim, 1)

        self.fm = FMLayer(n_dim=self.emb.emb_out_dim,k_dim=100)

        self.factor_estimating_net = MLP(hidden_dims=[self.emb.emb_out_dim]+deep_hidden_dims,dropouts=deep_dropouts)

        self.transform_p = nn.Linear(deep_hidden_dims[-1],len(deep_emb_inputs),bias=False)


    def forward(self,x):
        deep_inputs = x['deep']
        batch_size = deep_inputs.shape[0]
        cont_outs = deep_inputs[:, [self.deep_column_idx[col] for col in self.deep_continuous_cols]]
        emb_outs = self.emb(deep_inputs).to(torch.float32)

        dnn_outputs = self.factor_estimating_net(torch.flatten(emb_outs,start_dim=1))
        dnn_outputs = self.transform_p(dnn_outputs)
        input_aware_factor = len(self.deep_column_idx) * F.softmax(dnn_outputs,dim=1)

        #fm part
        fm_inputs = emb_outs * input_aware_factor.unsqueeze(-1)
        fm_outputs = self.fm(torch.flatten(fm_inputs,start_dim=1))

        #linear part
        linear_inputs = torch.cat([cont_outs, torch.flatten(emb_outs*input_aware_factor.unsqueeze(-1),start_dim=1)], dim=1).to(torch.float32)
        logits = self.linear(linear_inputs)
        logits = logits.add_(fm_outputs)

        return logits
