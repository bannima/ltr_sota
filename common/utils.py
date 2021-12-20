#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: utils.py
@time: 2021/12/20 4:46 PM
@desc: 
"""
import time

def which_day():
    return time.strftime('%Y-%m-%d',time.localtime(time.time()))
