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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
