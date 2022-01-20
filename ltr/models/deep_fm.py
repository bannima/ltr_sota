#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: deep_fm.py
@time: 2022/1/19 6:02 PM
@desc: 
"""
import torch
import torch.nn as nn
from ltr.models.layers.interaction import FMLayer
from ltr.models.layers.core import MLP,EmbAndConcat


class DeepFM(nn.Module):
    ''' DeepFM Networks '''
    def __init__(self,
                 deep_hidden_dims,
                 deep_dropouts,
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs,
                 k_dim=100
                 ):
        super(DeepFM, self).__init__()
        self.emb_cat = EmbAndConcat(deep_column_idx = deep_column_idx,deep_continuous_cols =deep_continuous_cols,deep_emb_inputs = deep_emb_inputs)
        self.fm = FMLayer(n_dim=self.emb_cat.out_dim,k_dim=k_dim)
        self.deep = MLP(hidden_dims=[self.emb_cat.out_dim]+deep_hidden_dims,dropouts=deep_dropouts)

    def forward(self, inputs):
        deep_inputs = inputs['deep']

        embed_outs, cont_outs = self.emb_cat(deep_inputs)
        deep_inputs = torch.cat([embed_outs, cont_outs], 1).to(torch.float32)

        fm_outs = self.fm(deep_inputs)
        fm_outs.add_(self.deep(deep_inputs))
        return fm_outs

