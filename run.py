#!usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function
import sys, os

sys.path.append(os.path.abspath(''))   # 把当前目录设为引用模块的地址之一

from utils import *
from data_utils import *
from models.solver_cnn import *
from models.ConvNet import *

import numpy as np
import pandas as pd
from itertools import product, permutations

import matplotlib.pyplot as plt

test_ctx()
print()


### Load Data ####
GW_address = './data/'

data = pd.DataFrame(np.load(GW_address+'GW_H1.npy'), index=np.load(GW_address+'GW_H1_index.npy'))
print(data.shape)
peak_samppoint = data.values.argmax(axis=1)
peak_samppoint = int(peak_samppoint.sum() / peak_samppoint.shape[0])
peak_time = peak_samppoint/data.shape[-1]
peak_time = float('{:.2f}'.format(peak_time))
print('Peak sampling point at %dth (%.2fs).' %(peak_samppoint, peak_time))
print()

### Split the Data
print('总波形数目：', data.index.shape)
train_masses = [(float(masses.split('|')[0]), float(masses.split('|')[1])) for masses in data.index if float(masses.split('|')[0]) % 2 != 0]
test_masses = [(float(masses.split('|')[0]), float(masses.split('|')[1])) for masses in data.index if float(masses.split('|')[0]) % 2 == 0]
print('训练集波形数目：', len(train_masses))
print('测试集波形数目：', len(test_masses))
print()

# 做好训练集和测试集的分割~
test_masses = [masses for masses in data.index if float(masses.split('|')[0]) % 2 == 0]
train_masses = [masses for masses in data.index if float(masses.split('|')[0]) % 2 != 0]
train_data = nd.array(data.loc[train_masses])
test_data = nd.array(data.loc[test_masses])

## Training
OURS_modified = ConvNet(conv_params = {'kernel': ((1,16), (1,8), (1,8), (1,8)), 
                                       'num_filter': (8, 16, 32, 64),
                                       'stride': ((1,1), (1,1), (1,1), (1,1)),
                                       'padding': ((0,0), (0,0), (0,0), (0,0)),
                                       'dilate': ((1,1), (1,1), (1,1), (1,1))},
                        act_params = {'act_type': ('elu', 'elu', 'elu', 'elu', 'elu', 'elu')},
                        pool_params = {'pool_type': ('max', 'max', 'max', 'max',),
                                       'kernel': ((1,8), (1,8), (1,8), (1,8), ),
                                       'stride': ((1,2), (1,2), (1,2), (1,2), ),
                                       'padding': ((0,0),(0,0), (0,0), (0,0), ),
                                       'dilate': ((1,1), (1,1), (1,1), (1,1), )},
                        fc_params = {'hidden_dim': (64,64)}, drop_prob = 0, 
#                         input_dim = (2,1,8192)
                        input_dim = (1,1,8192)
                       )
Solver = Solver_nd(model = OURS_modified, 
                   train = train,
                   test = test,
                   SNR = 1, #   params = params_tl,
                   num_epoch=10, 
                   batch_size = 256
                   ,  lr_rate=0.0003
                  ,save_checkpoints_address = './checkpoints/test/'
                  ,checkpoint_name = 'test')


Solver.Training()