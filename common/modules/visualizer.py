#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: visualizer.py
@time: 2021/12/21 1:43 PM
@desc: 
"""
import os
import matplotlib
import matplotlib.pyplot as plt

#chinese font file
font_file = os.path.join(os.path.dirname(__file__), 'project_file/HuaWenHeiTi-1.ttf')
zh_font = matplotlib.font_manager.FontProperties(
    fname = font_file
)

def draw_single_lines_chart(xticks,yticks,label,title="",save_path='./figs/',inter=None,rotation=30,fontsize=8):
    ''' 单条折线图 '''
    plt.clf()
    colors = ['blue','darkblue','red','orange','yellow']
    plt.plot(range(len(xticks)),yticks,linewidth=1.2,color=colors[0],label=label)
    if inter is None:
        inter = int(len(xticks)/6)
    plt.xticks(range(0,len(xticks),inter),xticks[0:len(xticks):inter],fontproperties=zh_font)
    plt.legend(prop=zh_font)
    plt.title(title,fontproperties=zh_font)
    plt.savefig(os.path.join(save_path,title),type='jpg',dpi=600)
    plt.show()


def draw_twin_lines_chart():
    ''' twin lines chart '''
    plt.cla()
    fig = plt.figure()

    ax1 = plt.add_subplot(111)
    ax1.set_ylim(0.78,1.0)


    ax2 = ax1.twinx()

