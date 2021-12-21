#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: mmoe.py
@time: 2021/12/20 4:41 PM
@desc: Implementation of MMoE: Modeling Task Relationship in Multi-task Learning with Multi-gate Mixture-of-Experts
"""
import torch
import torch.nn as nn

class Tower(nn.Module):
    ''' Task-specific tower network '''
    def __init__(self,input_size,hidden_size,output_size):
        super(Tower, self).__init__()
        self.f1 = nn.Linear(input_size,hidden_size)
        self.f1 = nn.Linear(hidden_size,output_size)

    def forward(self,inputs):
        outs = self.f1(inputs)
        return self.f2(outs)

class Expert(nn.Module):
    ''' Expert Network '''
    def __init__(self,input_size,hidden_size):
        super(Expert, self).__init__()
        self.linear = nn.Linear(input_size,hidden_size)

    def forward(self,inputs):
        return self.linear(inputs)

class MMoE(nn.Module):
    '''
    implementations of MMoE model
    '''
    def __init__(self, num_experts, num_task, input_size,hidden_size,output_size):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_task = num_task
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.experts = nn.ModuleList([Expert(self.input_size,hidden_size) for _ in range(self.num_experts)])

        self.gates = nn.ModuleList([nn.Linear(input_size,hidden_size) for _ in range(self.num_task)])

        self.towers = nn.ModuleList([Tower(input_size,hidden_size,output_size) for _ in range(self.num_task)])

    def forward(self, inputs):
        experts_output = torch.tensor([expert(inputs) for expert in self.experts])
        gate_logit = torch.tensor([gate(inputs) for gate in self.gates])
        #weighted sum
        outputs =  torch.bmm(experts_output,gate_logit)
        return [tower(outputs) for tower in self.towers]


