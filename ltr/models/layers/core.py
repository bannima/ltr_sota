#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: core.py
@time: 2022/1/5 5:42 PM
@desc: 
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.modules.data_utils import get_activation


def dense_layer(input_size,output_size,activation,dropout,batch_norm=True):
    ''' 生成dense layer '''
    act_func = get_activation(activation)
    linear = [nn.BatchNorm1d(input_size)] if batch_norm else[]
    linear += [nn.Linear(input_size,output_size),act_func,nn.Dropout(dropout)]
    return nn.Sequential(*linear)


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

class LocalActivationUnit(nn.Module):
    """
    The Local Activation Unit used in DIN with the representation of user interets varies adaptively
    given different candidate items.

    Input shape
        - A list of two 3D tensor with shape :``(batch_size,1,embedding_size)`` and  ``(batch_size,T,embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size,T,1)``.

    Arguments:

    """
    def __init__(self,hidden_units=(64,32),embedding_dim=4,activation='sigmod',\
                 dropout_rate=0,dice_dim=3,l2_reg=0,use_bn=False):
        super(LocalActivationUnit, self).__init__()
