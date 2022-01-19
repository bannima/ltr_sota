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
from ltr.models.layers.core import dense_layer

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

class EmbAndConcat(nn.Module):
    ''' category embedding layer and concatenate layer for deep model'''
    def __init__(self,deep_column_idx, deep_continuous_cols, deep_emb_inputs,dropout=0.3):
        super(EmbAndConcat, self).__init__()
        self.deep_column_idx = deep_column_idx
        self.deep_continuous_cols = deep_continuous_cols
        self.deep_emb_inputs = deep_emb_inputs
        #init category features embedding
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_{}".format(col):nn.Embedding(val+1,dim,padding_idx=0)
                for col,val,dim in self.deep_emb_inputs
            }
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.emb_out_dim = np.sum([embed[2] for embed in self.deep_emb_inputs])

        #init continuous features
        self.cont_out_dim = len(deep_continuous_cols)

        self.out_dim = self.emb_out_dim +self.cont_out_dim

    def forward(self,x):
        ''' given deep input x, return the concat of category embedding layer '''
        embed_outs = [
            self.embed_layers['emb_layer_{}'.format(col)](x[:,self.deep_column_idx[col]].long())
            for col,val,dim in self.deep_emb_inputs
        ]
        #concat by batch size dim
        embed_outs = torch.cat(embed_outs,dim=1)
        embed_outs = self.embedding_dropout(embed_outs)

        #continuous featrues
        cont_outs = x[:,[self.deep_column_idx[col] for col in self.deep_continuous_cols]]

        return embed_outs,cont_outs

class MLP(nn.Module):
    ''' basic deep model '''
    def __init__(self, hidden_dims, dropouts,deep_column_idx,deep_continuous_cols, deep_emb_inputs):
        super(MLP,self).__init__()
        assert isinstance(hidden_dims, list)

        #category features embedding and concat
        self.emb_cat = EmbAndConcat(deep_column_idx,deep_continuous_cols, deep_emb_inputs)
        #features should embedding first
        self.hidden_dims = [self.emb_cat.out_dim]+hidden_dims
        self.mlp = nn.Sequential()
        for i in range(1,len(self.hidden_dims)):
            self.mlp.add_module(
                "dense_layer_{}".format(i-1),
                dense_layer(self.hidden_dims[i-1],
                            self.hidden_dims[i],
                            activation='relu',
                            dropout=dropouts[i-1])
            )
        for name,tensor in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor,mean=0,std=0.0001)

    def forward(self,x):
        #transform to category embed and continuous out
        embed_outs,cont_outs = self.emb_cat(x)
        outs = torch.cat([embed_outs, cont_outs], 1).to(torch.float32)
        return self.mlp(outs)

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
        self.deep = MLP(hidden_dims=deep_hidden_dims,
                        dropouts=deep_dropouts,
                        deep_column_idx = deep_column_idx,
                        deep_continuous_cols =deep_continuous_cols,
                        deep_emb_inputs = deep_emb_inputs
                        )

    def forward(self, inputs):
        wide_inputs = inputs['wide']
        deep_inputs = inputs['deep']
        wide_outs = self.wide(wide_inputs)
        wide_outs.add_(self.deep(deep_inputs))
        return wide_outs
