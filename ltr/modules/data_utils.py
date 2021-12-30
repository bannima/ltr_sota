#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: data_utils.py
@time: 2021/12/21 11:09 AM
@desc: 
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

def get_activation(activation):
    if activation =='relu':
        return nn.ReLU(inplace=True)
    elif activation=='leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif activation=='tanh':
        return nn.Tanh()
    elif activation=='gelu':
        return nn.GELU()
    elif activation=='geglu':
        return nn.GEGLU()
    elif activation=='reglu':
        return nn.REGLU()
    elif activation=='softplus':
        return nn.Softplus()
    else:
        raise ValueError("{} activation not recognized".format(activation))


def convert_to_bert_batch_data(data, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for row in data:
        encoded_dict = tokenizer.encode_plus(
            row,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        try:
            token_type_ids.append(encoded_dict['token_type_ids'])
        except:
            pass
    # convert list to tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if len(token_type_ids) != 0:
        token_type_ids = torch.cat(token_type_ids, dim=0)
        return TensorDataset(input_ids, attention_masks, token_type_ids)
    else:
        return TensorDataset(input_ids, attention_masks)
