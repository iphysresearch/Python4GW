#!usr/bin/python
#coding=utf-8

import sys, os
sys.path.append(os.path.abspath(''))   # 把当前目录设为引用模块的地址之一

from utils import *
from data_utils import *
from models.solver_cnn import *
from models.ConvNet import *

import numpy as np
import pandas as pd
from itertools import product, permutations
from sklearn import metrics

test_ctx()


### Load Data ####
#GW_address = '/floyd/input/waveform/'
GW_address = './data/'

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

MODEL = 'OURs_modified'
#pretrained_add = '/floyd/input/pretrained/pretrained_models/%s/' %MODEL
pretrained_add = './pretrained_models/%s/' %MODEL
os.system('ls -a %s | grep best > test.txt' %pretrained_add)
params_adds = pd.read_csv('./test.txt', header=None)
os.system('rm test.txt')
params_adds['snr'] = params_adds[0].map(lambda x: int(x.split('_')[1]))
params_adds = params_adds.sort_values('snr', ascending=False)[0].values.tolist()

print(params_adds[-1])

auc_frame = []

param = nd.load(pretrained_add + params_adds[-1])
    
OURs_modified = ConvNet(conv_params = {'kernel': ((1,16), (1,8), (1,8)), 
                            'num_filter': (32, 64, 128,),
                            'stride': ((1,1), (1,1), (1,1),),
                            'padding': ((0,0), (0,0), (0,0),),
                            'dilate': ((1,1), (1,1), (1,1),)},
                act_params = {'act_type': ('elu', 'elu', 'elu', 'elu',)},
                pool_params = {'pool_type': ('max', 'max', 'max',),
                            'kernel': ((1,4), (1,4), (1,4),),
                            'stride': ((1,2), (1,2), (1,2),),
                            'padding': ((0,0),(0,0), (0,0),),
                            'dilate': ((1,1), (1,1), (1,1),)},
                fc_params = {'hidden_dim': (256,)}, drop_prob = 0, 
#                         input_dim = (2,1,8192)
                input_dim = (1,1,8192)
                    )
auc_list = []
snr_list = np.linspace(0.1, 1, 10)
j = 0

Solver = Solver_nd(model = OURs_modified, 
                train = train_data,
                test = test_data,
                SNR = 1, 
                batch_size = 10)

auc_var_list = []
i = 0            
prob, label , _= Solver.predict_nd()
print(prob[:10])
print(label[:10])    

