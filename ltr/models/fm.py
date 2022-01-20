#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: fm.py
@time: 2022/1/18 4:50 PM
@desc: Implementation of Factorization Machine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltr.models.layers.core import EmbAndConcat
from ltr.models.layers.interaction import FMLayer



class FM(nn.Module):
    ''' Factorization Machine with feature preprocess '''
    def __init__(self,
                 hidden_vector_dim,
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs,
                 ):
        super(FM, self).__init__()
        self.emb_cat = EmbAndConcat(deep_column_idx,deep_continuous_cols, deep_emb_inputs)

        self.fm = FMLayer(self.emb_cat.out_dim,hidden_vector_dim)

    def forward(self, inputs):
        deep_inputs = inputs['deep']
        embed_outs, cont_outs = self.emb_cat(deep_inputs)
        deep_inputs = torch.cat([embed_outs, cont_outs], 1).to(torch.float32)
        return self.fm(deep_inputs)






