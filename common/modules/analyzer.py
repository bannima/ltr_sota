#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: analyzer.py
@time: 2021/12/25 3:59 PM
@desc: 
"""
from abc import ABCMeta,abstractmethod
import pandas as pd
from common.modules.visualizer import draw_single_lines_chart

class ExperimentAnalyzer(metaclass=ABCMeta):
    ''' Analysis the whole experiment '''
    def __init__(self,stats_file):
        self.statistics = pd.read_csv(stats_file, sep=',', encoding='utf-8')

    @abstractmethod
    def analysis_experiment(self):
        raise NotImplementedError

class SingleTaskExpAnalyzer(ExperimentAnalyzer):
    def __init__(self,stats_file):
        super(SingleTaskExpAnalyzer, self).__init__(stats_file)

    def analysis_experiment(self):
        #load experiment statistics
        pass

class MultiTaskExpAnalyzer(ExperimentAnalyzer):
    ''' Analysis the Multi task Experiment '''
    def __init__(self,stats_file):
        super(MultiTaskExpAnalyzer, self).__init__(stats_file)

    def analysis_experiment(self):
        #load experiment statistics
        pass








