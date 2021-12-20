#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: config.py
@time: 2021/12/20 4:42 PM
@desc: 
"""
import os
import logzero
from logzero import logger

from common.utils import which_day

#project path
project_path = os.path.dirname(os.path.dirname(__file__))
#dataset path
dataset_path = os.path.join(project_path,'dataset')

#logger
log_path = os.path.join(project_path,'logs')
log_file = os.path.join(log_path,"{}_logfile.log".format(which_day()))
logzero.logfile(log_file,maxBytes=1e6,backupCount=3)



