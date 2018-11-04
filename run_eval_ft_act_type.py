#!usr/bin/python
#coding=utf-8

import sys, os
sys.path.append(os.path.abspath(''))   # 把当前目录设为引用模块的地址之一

from utils import *
from data_utils import *
from models.solver_cnn_ import *
from models.ConvNet import *

import numpy as np
import pandas as pd
from itertools import product, permutations
from sklearn import metrics

test_ctx()


def Fine_tune(name, value):
    default = {'default':{'drop_prob': 0,
                          'fc_params_act_type': ({'hidden_dim':(64,)},
                                                 {'act_type': ('relu',)*4}),
                          'pool_type_kernel': (('avg','avg','avg',), 
                                               ((1,16), (1,16), (1,16),)),
                          'dialute': 1,
                          'num_filter': (16,32,64) }}
    df = pd.DataFrame(default)
    df.drop([name], inplace=True)
    params_list = []
    for i,j in product(value, dict(df).values()):
        dd = dict(j)
        dd[name] = i
        dd['hidden_dim'] = dd['fc_params_act_type'][0]
        dd['act_type'] = dd.pop('fc_params_act_type')[1]
        dd['pool_type'] = dd['pool_type_kernel'][0]
        dd['pool_kernel'] = dd.pop('pool_type_kernel')[1]        
        params_list.append(dd)
    return params_list
def test(diedai):
    for hyperparam in diedai:
        hidden_dim = hyperparam['hidden_dim']
        drop_prob = hyperparam['drop_prob']
        num_filter = hyperparam['num_filter']
        act_type= hyperparam['act_type']
        dialute= hyperparam['dialute']
        pool_kernel = hyperparam['pool_kernel']
        pool_type = hyperparam['pool_type']
        print()
        print('hidden_dim |', hidden_dim)
        print('drop_prob |' , drop_prob)
        print('num_filter |', num_filter)
        print('act_type |' , act_type)
        print('dialute |', dialute)
        print('pool_kernel |', pool_kernel)
        print('pool_type |', pool_type)


### Load Data ####
GW_address = '/floyd/input/waveform/'
# GW_address = './data/'
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


MODEL = 'OURs_new_ft_act_type'
pretrained_add = '/floyd/input/pretrained/pretrained_models/OURs_fine_tune/%s/' %MODEL
# pretrained_add = './pretrained_models/OURs_finetune/OURs_new_ft_act_type'
os.system('ls -a %s | grep best > test.txt' %(pretrained_add))
params_adds = pd.read_csv('./test.txt', header=None)
os.system('rm test.txt')
params_adds['drop_prob'] = params_adds[0].map(lambda x: (x.split('_')[2]))
params_adds = params_adds.sort_values('drop_prob', ascending=False)[0].values.tolist()  #['relu','elu']

print(params_adds)

auc_frame = []
for param_add, hyperparam in zip(params_adds,  Fine_tune('fc_params_act_type', [({'hidden_dim': (64,)}, {'act_type': ('relu',)*4}),
                                                                                 ({'hidden_dim': (64,)}, {'act_type': ('elu',)*4})])):

    print('Now working on:')
    # test(Fine_tune('fc_params_act_type', [({'hidden_dim': (64,)}, {'act_type': ('elu',)*4})]))
    print(hyperparam)
    print(param_add)
    param = nd.load(pretrained_add + param_add)

    hidden_dim = hyperparam['hidden_dim']
    drop_prob = hyperparam['drop_prob']
    num_filter = hyperparam['num_filter']
    act_type= hyperparam['act_type']
    dialute= hyperparam['dialute']
    pool_kernel = hyperparam['pool_kernel']
    pool_type = hyperparam['pool_type']
    
    OURS_ori = ConvNet(conv_params = {'kernel': ((1,16), (1,8), (1,8)), 
                                        'num_filter': num_filter,
                                        'stride': ((1,1), (1,1), (1,1),),
                                        'padding': ((0,0), (0,0), (0,0),),
                                        'dilate': ((1,dialute), (1,dialute), (1,dialute),)},
                       act_params = act_type,
                       pool_params = {'pool_type': pool_type,
                                        'kernel': pool_kernel,
                                        'stride': ((1,2), (1,2), (1,2),),
                                        'padding': ((0,0),(0,0), (0,0),),
                                        'dilate': ((1,1), (1,1), (1,1),)},
                       fc_params = hidden_dim, drop_prob = drop_prob,
                       params_inits = param,
                       input_dim = (1,1,8192)
                      )
    auc_list = []
    snr_list = np.linspace(0.1, 1, 10)
    j = 0
    while True:
        try:
            snr = snr_list[j]
        except IndexError:
            break
        print('SNR = %s' %snr)
        try:
            Solver = Solver_nd(model = OURS_ori, 
                            train = train_data,
                            test = test_data,
                            SNR = snr, 
                            batch_size = 256)
        except mx.MXNetError as e:
            print(e)
            print('Rerunning...')
            continue
        auc_var_list = []
        i = 0
        while True:
            if i == 2: break
            else: pass
            try:
                prob, label , _= Solver.predict_nd()
            except mx.MXNetError as e:
                print(e)
                print('Rerunning...')
                continue
            fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auc_var_list.append(auc)
            print('{"metric": "AUC for SNR(model,test)=(%s,(0.1~10))", "value": %.5f}' %(param_add.split('_')[2], auc) )
            i += 1
        j += 1
        
        auc_list.append(auc_var_list)
    auc_frame.append(auc_list)
os.system('rm -rf ./*')
np.save('./AUC_%s' %MODEL, np.array(auc_frame))

# floyd run --gpu \
# --data wctttty/datasets/gw_waveform/1:waveform \
# --data wctttty/projects/python4gw/197:pretrained \
# -m "AUC_new_ft_act_type" \
# "bash setup_floydhub.sh && python run_eval_ft_act_type.py"