#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: wide_deep.py
@time: 2021/12/27 5:45 PM
@desc: Implementations of  "Wide & Deep Learning for Recommender Systems"
"""
import math
import torch
import torch.nn as nn
import numpy as np
from ltr.modules.data_utils import get_activation
from ltr.models.layers.core import dense_layer,EmbAndConcat,MLP

class Wide(nn.Module):
    def __init__(self, wide_dim: int, pred_dim: int = 1):
        super(Wide, self).__init__()
        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.wide_linear = nn.Embedding(wide_dim + 1, pred_dim, padding_idx=0)
        # (Sum(Embedding) + bias) is equivalent to (OneHotVector + Linear)
        self.bias = nn.Parameter(torch.zeros(pred_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        r"""initialize Embedding and bias like nn.Linear. See `original
        implementation
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear>`_.
        """
        nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):  # type: ignore
        r"""Forward pass. Simply connecting the Embedding layer with the ouput
        neuron(s)"""
        out = self.wide_linear(inputs.long()).sum(dim=1) + self.bias
        return out

class WideDeep(nn.Module):
    ''' Wide & Deep Networks '''
    def __init__(self,
                 wide_dim,
                 pred_dim,
                 deep_hidden_dims,
                 deep_dropouts,
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs
                 ):
        super(WideDeep, self).__init__()
        self.wide = Wide(wide_dim=wide_dim,pred_dim=pred_dim)

        # category features embedding and concat
        self.emb_cat = EmbAndConcat(deep_column_idx, deep_continuous_cols, deep_emb_inputs)
        self.deep = MLP(hidden_dims=[self.emb_cat.out_dim]+deep_hidden_dims,
                        dropouts=deep_dropouts)

    def forward(self, inputs):
        wide_inputs = inputs['wide']
        deep_inputs = inputs['deep']
        wide_outs = self.wide(wide_inputs)

        # transform to category embed and continuous out
        embed_outs, cont_outs = self.emb_cat(deep_inputs)
        deep_inputs = torch.cat([embed_outs, cont_outs], 1).to(torch.float32)

        wide_outs.add_(self.deep(deep_inputs))
        return wide_outs
