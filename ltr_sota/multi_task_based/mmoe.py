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
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmod = nn.Sigmoid()

    def forward(self,inputs):
        outs = self.fc1(inputs)
        outs = self.relu(outs)
        outs = self.dropout(outs)
        outs = self.fc2(outs)
        outs = self.sigmod(outs)
        return outs

class Expert(nn.Module):
    ''' Expert Network '''
    def __init__(self,input_size,hidden_size,output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self,inputs):
        outs = self.fc1(inputs)
        outs = self.relu(outs)
        outs = self.dropout(outs)
        outs = self.fc2(outs)
        return outs

class MMoE(nn.Module):
    '''
    implementations of MMoE model
    '''
    def __init__(self,input_size, num_experts, experts_out, experts_hidden, towers_hidden,output_size, tasks):
        super(MMoE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.output_size = output_size
        self.tasks = tasks

        self.softmax=nn.Softmax(dim=1)

        self.experts = nn.ModuleList([Expert(self.input_size,self.experts_hidden,self.experts_out) for _ in range(self.num_experts)])
        self.gates = nn.ParameterList([nn.Parameter(torch.randn(self.input_size,self.num_experts),requires_grad=True) for _ in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out,self.towers_hidden,self.output_size) for _ in range(self.tasks)])

    def forward(self, inputs):
        #
        experts_outs = torch.stack([expert(inputs) for expert in self.experts])
        gates_o = [self.softmax(inputs @ gate) for gate in self.gates]

        tower_input = [gate_o.t().unsqueeze(2).expand(-1,-1,self.experts_out)*experts_outs for gate_o in gates_o ]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        final_output = [tower(ti) for tower, ti in zip(self.towers, tower_input)]
        return final_output

