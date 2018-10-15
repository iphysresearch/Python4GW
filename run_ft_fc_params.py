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
print()
test_ctx()
print()


### Load Data ####
GW_address = '/floyd/input/waveform/'

data = pd.DataFrame(np.load(GW_address+'GW_H1.npy'), index=np.load(GW_address+'GW_H1_index.npy'))
print('Raw data: ', data.shape)
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
train_data = nd.array(data.loc[train_masses], ctx=mx.cpu())
test_data = nd.array(data.loc[test_masses], ctx=mx.cpu())

## Training

# for snr in list([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):

# params_tl  = nd.load('/floyd/input/pretrained/OURs/snr_8_best_params_epoch@16.pkl')
# for snr in list([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
# params_tl  = nd.load('/floyd/input/pretrained/OURs/snr_3_best_params_epoch@3.pkl')
SNR_list = [1, 0.6, 0.4, 0.3, 0.2]
fc_params_list = [{'hidden_dim': (64,64,)},
             {'hidden_dim': (64,64,64,)},
             {'hidden_dim': (64,64,64,64,)}]
act_params_list = [{'act_type': ('relu', 'relu', 'relu', 'relu','relu')},
                   {'act_type': ('relu', 'relu', 'relu', 'relu','relu','relu')},
                   {'act_type': ('relu', 'relu', 'relu', 'relu','relu','relu','relu')}]

for index , (fc_params, act_params) in enumerate(zip(fc_params_list, act_params_list)):
    print('fc_params:' , fc_params)
    print('act_params:' , act_params)
    i = 0
    params_tl  = None
    while True:
        try:
            snr = SNR_list[i]
        except IndexError:
            break

        OURS_ori = ConvNet(conv_params = {'kernel': ((1,16), (1,8), (1,8)), 
                                            'num_filter': (16, 32, 64,),
                                            'stride': ((1,1), (1,1), (1,1),),
                                            'padding': ((0,0), (0,0), (0,0),),
                                            'dilate': ((1,1), (1,1), (1,1),)},
                                act_params = act_params,
                                pool_params = {'pool_type': ('avg', 'avg', 'avg',),
                                            'kernel': ((1,16), (1,16), (1,16),),
                                            'stride': ((1,2), (1,2), (1,2),),
                                            'padding': ((0,0),(0,0), (0,0),),
                                            'dilate': ((1,1), (1,1), (1,1),)},
                                fc_params = fc_params, drop_prob = 0, 
        #                         input_dim = (2,1,8192)
                                input_dim = (1,1,8192)
                            )

        Solver = Solver_nd(model = OURS_ori, 
                        train = train_data,
                        test = test_data,
                        SNR = snr,   params = params_tl,
                        num_epoch=30, 
                        batch_size = 256
                        ,  lr_rate=0.0003
                        ,save_checkpoints_address = './OURs/'
                        ,checkpoint_name = 'fc_params_%s' %int(index+2),verbose =True, )

        try:
            Solver.Training()
        except mx.MXNetError:
            print('Rerunning...')
            continue

        params_tl = Solver.best_params
        i += 1