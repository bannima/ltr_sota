#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: __init__.py.py
@time: 2021/12/20 4:08 PM
@desc: 
"""
from ltr.models.fm import FM
from ltr.models.mmoe import MMoE
from ltr.models.wide_deep import WideDeep
from ltr.models.deep_cross import DeepCross
from ltr.models.deep_fm import DeepFM
from ltr.models.nfm import NFM
from ltr.models.afm import AFM

__all__ = [
    'FM',
    'MMoE',
    'WideDeep',
    'DeepCross',
    'DeepFM',
    'NFM',
    'AFM'
]
