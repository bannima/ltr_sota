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
import argparse
import datetime
import json
import os.path
import time

from tqdm import tqdm


def parse_parmas():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, required=False, default=64)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--epoch', type=int, required=False, default=10)
    parser.add_argument('--criterion', type=str, required=False, default='BCELoss')
    parser.add_argument('--metrics', type=str, required=False, default='classification')
    parser.add_argument('--save_model', type=bool, required=False, default=True)
    parser.add_argument('--save_test_preds', type=bool, required=False, default=True)

    args = parser.parse_args()
    HYPERS = {
        'LearningRate': args.lr,
        'Epochs': args.epoch,
        'Batch': args.batchsize,
        'Criterion': args.criterion,
        'Metrics': args.metrics,
        'Save_Model':args.save_model,
        'save_test_preds':args.save_test_preds
    }

    return HYPERS


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def which_day():
    return time.strftime('%Y-%m-%d', time.localtime(time.time()))


def current_time():
    return time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))


def save_to_json(datas, filepath, type='w'):
    with open(filepath, type, encoding='utf-8') as fout:
        for data in datas:
            json.dump(data, fout, ensure_ascii=False)
            fout.write('\n')


def read_from_json(filepath):
    if not os.path.exists(filepath):
        raise RuntimeError(" File not exists for {}".format(filepath))
    dataframe = []
    with open(filepath, 'r', encoding='utf-8') as fread:
        for line in tqdm(fread.readlines(), desc=' read json file', unit='line'):
            dataframe.append(json.loads(line.strip('\n')))
    return dataframe


def flatten(lists):
    ''' flatten nested lists-like object '''
    for x in lists:
        if hasattr(x,'__iter__') and not isinstance(x,(str,bytes)):
            yield from flatten(x)
        else:
            yield x