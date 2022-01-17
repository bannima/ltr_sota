#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: deep_cross.py
@time: 2022/1/2 9:09 PM
@desc:  Implementation of 《 Deep & Cross Network for Ad Click Predictions 》
"""

import math
import torch
import torch.nn as nn
import numpy as np
from ltr.modules.data_utils import get_activation

def dense_layer(input_size,output_size,activation,dropout):
    ''' 生成dense layer '''
    act_func = get_activation(activation)
    linear = [nn.Linear(input_size,output_size),act_func,nn.Dropout(dropout)]
    return nn.Sequential(*linear)

class CrossLayer(nn.Module):
    ''' Cross Layer '''
    def __init__(self,layer_size,activation,dropout):
        super(CrossLayer, self).__init__()
        self.act_func = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(layer_size,eps=1e-8)

        self.weight = nn.Parameter(torch.zeros(layer_size,1))
        #nn.init.normal_(self.weight, mean=0, std=0.0001)
        self.bias = nn.Parameter(torch.zeros(layer_size,1))

    def forward(self,x):
        x,x0 = x
        x = self.bn(x)
        out = torch.bmm(x0.unsqueeze(dim=2),x.unsqueeze(dim=2).transpose(-1,-2))
        out = torch.matmul(out,self.weight)
        out.add_(self.bias)
        out = out.squeeze(dim=2)
        out.add_(x)
        return out

class CrossNetwork(nn.Module):
    ''' Cross Network '''
    def __init__(self,
                 input_dim, #input dim for cross layer
                 cross_layers, # num of layers for cross networks
                 cross_dropouts, # dropout for cross network
                ):
        super(CrossNetwork, self).__init__()

        # features should embedding first
        self.cross = nn.Sequential()
        for i in range(cross_layers):
            self.cross.add_module(
                "cross_layer_{}".format(i),
                CrossLayer( input_dim,
                            activation='relu',
                            dropout=cross_dropouts[i])
            )
        for name,tensor in self.cross.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor,mean=0,std=0.0001)

    def forward(self, x):
        x0 = x.clone()
        for cross_layer in self.cross:
            x = cross_layer([x,x0])
        return x

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

        #category features embedding and concat
        #features should embedding first
        self.hidden_dims = hidden_dims
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

        return self.mlp(x)

class DeepCross(nn.Module):
    ''' Deep & Cross Networks '''
    def __init__(self,
                 cross_input_dim, #input dim for cross network
                 cross_layers, # num of layers for cross networks
                 cross_dropouts, # dropout for cross network
                 deep_input_dim, # input dim for deep layer
                 deep_layers,  # num of layers for deep networks
                 deep_dropouts, # dropout for deep network
                 deep_column_idx,
                 deep_continuous_cols,
                 deep_emb_inputs,
                 deep_dims
                 ):
        super(DeepCross, self).__init__()
        self.emb_cat = EmbAndConcat(deep_column_idx,deep_continuous_cols, deep_emb_inputs)

        self.cross = CrossNetwork(
                 input_dim=cross_input_dim, #input dim for cross layer
                 cross_layers=cross_layers, # num of layers for cross networks
                 cross_dropouts=cross_dropouts, # dropout for cross network
                )
        self.deep = MLP(
                hidden_dims=[deep_input_dim]*deep_layers,
                dropouts=deep_dropouts,
                deep_column_idx=deep_column_idx,
                deep_continuous_cols=deep_continuous_cols,
                deep_emb_inputs=deep_emb_inputs
                        )

        self.combination = nn.Linear((cross_input_dim+deep_input_dim),1)

    def forward(self, inputs):
        deep_inputs = inputs['deep']
        embed_outs, cont_outs = self.emb_cat(deep_inputs)
        deep_inputs = torch.cat([embed_outs, cont_outs], 1).to(torch.float32)

        cross_outs = self.cross(deep_inputs)
        deep_outs = self.deep(deep_inputs)
        outs = torch.cat((deep_outs,cross_outs),dim=1).to(torch.float32)
        logits = self.combination(outs)
        return logits
