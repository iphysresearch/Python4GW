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

MODEL = 'PRL'
pretrained_add = '/floyd/input/pretrained/pretrained_models/%s/' %MODEL
os.system('ls -a %s | grep best > test.txt' %pretrained_add)
params_adds = pd.read_csv('./test.txt', header=None)
os.system('rm test.txt')
params_adds['snr'] = params_adds[0].map(lambda x: int(x.split('_')[1]))
params_adds = params_adds.sort_values('snr', ascending=True)[0].values.tolist()

print(params_adds)

fpr_frame = []
tpr_frame = []
auc_frame = []
for param_add in params_adds:
    print('Now working on  %s' %param_add)
    param = nd.load(pretrained_add + param_add)
    
    PRL = ConvNet(conv_params = {'kernel': ((1,64), (1,32), (1,32), (1,16),(1,16),(1,16)), 
                                'num_filter': (8, 8, 16, 16, 32, 32),
                                'stride': ((1,1), (1,1), (1,1),(1,1),(1,1),(1,1),),
                                'padding': ((0,0), (0,0), (0,0),(0,0),(0,0),(0,0),),
                                'dilate': ((1,1), (1,1), (1,1),(1,1),(1,1),(1,1),)},
                    act_params = {'act_type': ('elu', 'elu', 'elu', 'elu','elu','elu','elu','elu')},
                    pool_params = {'pool_type': ('max', 'max', 'max','max','max','max',),
                                'kernel': ((1,1), (1,8), (1,1),(1,6),(1,1),(1,4)),
                                'stride': ((1,2), (1,2), (1,2),(1,2),(1,2),(1,2)),
                                'padding': ((0,0),(0,0), (0,0), (0,0),(0,0),(0,0)),
                                'dilate': ((1,1), (1,1), (1,1),(1,1),(1,1),(1,1))},
                    fc_params = {'hidden_dim': (64,64)}, drop_prob = 0.5, 
                    params_inits = param,
                    input_dim = (1,1,8192)
                    )

    auc_list = []
    fpr_list = []
    tpr_list = []
    snr_list = np.linspace(0.1, 1, 10)
    # snr_list = (np.array([0.01]+np.arange(0.05,1.04,0.05).tolist()))[::-1]
    j = 0
    while True:
        try:
            snr = snr_list[j]
            print('Testing for snr=', snr)
        except IndexError:
            break

        try:
            Solver = Solver_nd(model = PRL, 
                            train = train_data,
                            test = test_data,
                            SNR = snr, 
                            batch_size = 256)
        except mx.MXNetError:
            print('Rerunning...')
            continue
        auc_var_list = []
        fpr_var_list, tpr_var_list = [], []
        i = 0
        while True:
            if i == 5: break
            else: pass
            try:
                prob, label , _= Solver.predict_nd()
            except mx.MXNetError:
                print('Rerunning...')
                continue
            fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auc_var_list.append(auc)
            fpr_var_list.append(fpr)
            tpr_var_list.append(tpr)            
            print('{"metric": "AUC for SNR(model,test)=(%s,(0.1~10))", "value": %.5f}' %(param_add.split('_')[1], auc) )
            i += 1
        j += 1

        auc_list.append(auc_var_list)
        fpr_list.append(fpr_var_list)
        tpr_list.append(tpr_var_list)
    auc_frame.append(auc_list)
    fpr_frame.append(fpr_list)
    tpr_frame.append(tpr_list)

np.save('./AUC_%s' %MODEL, np.array(auc_frame))
np.save('./fpr_%s' %MODEL, np.array(fpr_frame))
np.save('./tpr_%s' %MODEL, np.array(tpr_frame))    
# np.save('./AUC_%s' %MODEL, np.array(auc_frame))


# floyd run --gpu \
# --data wctttty/datasets/gw_waveform/1:waveform \
# --data wctttty/projects/python4gw/159:pretrained \
# -m "AUC_PRL_new" \
# "bash setup_floydhub.sh && python run_eval_PRL.py"