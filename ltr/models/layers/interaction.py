#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: interaction.py
@time: 2022/1/5 5:42 PM
@desc: 
"""
import torch
import torch.nn as nn
from itertools import combinations
import torch.nn.functional as F

class FMLayer(nn.Module):
    """Factorization Machines

    Input shape:
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
    Output shape:
        - 2D tensor with shape: ``(batch_size,1)``
    References
        - [Factorization Machines](www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

    """

    def __init__(self,n_dim,k_dim):
        '''
        :param n_dim: feature size
        :param k_dim: dimension of hidden vector V
        '''
        super(FMLayer, self).__init__()
        self.n_dim = n_dim
        self.k_dim = k_dim
        #linear layer
        self.linear = nn.Linear(self.n_dim,1)
        #interaction matrix
        self.V = nn.Parameter(torch.randn(self.n_dim,self.k_dim))
        nn.init.normal_(self.V,mean=0,std=1e-3)

    def forward(self,x):
        '''
        :param x: 2D tensor with shape: ``(batch_size,feature_size)``
        :return: 2D tensor with shape: ``(batch_size,1)``
        '''

        # 1. traditional way
        # linear_outs = self.linear(x) # B x 1
        # v_v = torch.mm(self.V,self.V.T) # (N x K) @ (K x N) = N x N
        # x = x.unsqueeze(dim=1) # B x 1 x N
        # x_x = torch.bmm(x.transpose(-1,-2),x) # (B x N x 1) @ (B x 1 x N) = B x N x N
        # interaction_outs = torch.matmul(x_x,v_v) # N x N, element-wise product
        # linear_outs.add_(torch.sum(interaction_outs,dim=(1,2),keepdim=False).unsqueeze(dim=1)) # (B x N x N) -> (B x 1 x 1) -> B -> (B x 1)
        # return linear_outs

        # 2. ugly way
        # batch_size = x.shape[0]
        # linear_outs = self.linear(x) # B x 1
        #
        # sum_of_square = torch.pow(torch.matmul(x.unsqueeze(dim=1),self.V).squeeze(dim=1),2)
        # sum_of_square = 0.5*torch.sum(sum_of_square,dim=1,keepdim=True)
        #
        # square_of_sum = torch.matmul(x.unsqueeze(dim=2).expand(batch_size,self.n_dim,self.n_dim),self.V)
        # square_of_sum = torch.bmm(square_of_sum,square_of_sum.transpose(-1,-2))
        # square_of_sum = -0.5 * torch.sum(square_of_sum,dim=(1,2),keepdim=True).squeeze(dim=2)
        #
        # linear_outs.add_(sum_of_square)
        # linear_outs.add_(square_of_sum)
        # return linear_outs

        # 3.elegant way
        x1 = self.linear(x)
        square_of_sum = torch.mm(x,self.V) * torch.mm(x,self.V)
        sum_of_square = torch.mm(x*x,self.V*self.V)
        x2 = 0.5 * torch.sum((square_of_sum-sum_of_square),dim=-1,keepdim=True)
        x = x1 + x2
        return x

class BiInteractionLayer(nn.Module):
    """ Bi-Interaction Layer in Neural Factorization Machines """
    def __init__(self):
        super(BiInteractionLayer, self).__init__()

    def forward(self,x):
        """
        Input Shape
            - A 3D tensor with shape: ``(batch_size,feature_size,embedding_size)``

        Output Shape
            - 2D tensor with shape: ``(batch_size,embedding_size)``
        """
        square_of_sum = torch.pow(
            torch.sum(x,dim=1,keepdim=True),2
        )
        sum_of_square = torch.sum(
            torch.pow(x,2),dim=1,keepdim=True
        )
        x = 0.5*(square_of_sum-sum_of_square).squeeze(dim=1)
        return x

class AFMLayer(nn.Module):
    """Attentional Factorization Machines Layer"""
    def __init__(self,embedding_size,attention_factor=10,dropout=0.3):
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.dropout = nn.Dropout(dropout)

        self.attention_W = nn.Parameter(
            torch.Tensor(embedding_size,self.attention_factor)
        )

        self.attention_b = nn.Parameter(
            torch.Tensor(self.attention_factor)
        )

        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor,1))

        self.projection_p = nn.Parameter(torch.Tensor(embedding_size,1))

        for tensor in [self.attention_W,self.projection_h,self.projection_p]:
            nn.init.xavier_normal_(tensor)

        for tensor in self.attention_b:
            nn.init.zeros_(tensor)

    def forward(self,x):
        """
        Input Shape
            - A 3D tensor with shape: ``(batch_size,feature_size,embedding_size)``

        Output Shape
            - 2D tensor with shape: ``(batch_size,1)``
        """
        batch_size,feature_size,embedding_size = x.shape
        p = []
        q = []
        #feature combinations
        for i,j in combinations(list(range(feature_size)),2):
            p.append(x[:,i,:])
            q.append(x[:,j,:])
        p = torch.stack(p,dim=1)
        q = torch.stack(q,dim=1)
        inter_product = torch.mul(p,q) # batch_size X combination_feature_size X embedding_size
        attention_tmp = F.relu(
            torch.tensordot(inter_product,self.attention_W,dims=([2],[0]))+self.attention_b
        ) # batch_size X combination_feature_size X attention_factor
        norm_attention_score = F.softmax(
            torch.tensordot(attention_tmp,self.projection_h,dims=([2],[0])),dim=1
        ) #batch_size X combination_feature_size X 1
        attention_out = torch.sum(
            norm_attention_score * inter_product,dim=1
        )#batch_size,embedding_size
        attention_out = self.dropout(attention_out)
        #afm_out = torch.tensordot(attention_out,self.projection_p,dims=([1],[0]))
        afm_out = torch.matmul(attention_out,self.projection_p)
        return afm_out

