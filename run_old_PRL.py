#!usr/bin/python
#coding=utf-8


# importing the library
from __future__ import print_function
import pandas as pd
import numpy as np
import seaborn as sns

from scipy import signal
import scipy
import math
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.gridspec import GridSpec

import time
import os, sys


import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random
from data_utils import *

mx.random.seed(1)
random.seed(1)

try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
print('CPU or GPU? : ', ctx)



# zero mean and unit variance as it makes traning process easier
def Normolise(data):
    data_array = np.array(data)
    data_array_shape = data_array.shape[0]
    return pd.DataFrame((data_array -np.mean(data_array, axis=1).reshape(data_array_shape,-1))/np.std(data_array, axis=1).reshape(data_array_shape,-1)
                        ,index = data.index)

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃。
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape, ctx=ctx) < keep_prob
    return mask * X / keep_prob

def init_params(num_fc1, num_fc2, num_outputs, sl):
    #######################
    #  Set the scale for weight initialization and choose
    #  the number of hidden units in the fully-connected layer
    #######################
    weight_scale = .01

    # P1 = np.sqrt(6)/np.sqrt(256)
    # W1 = nd.random_uniform(low = -P1, high= P1, shape=(16, 1, 1, 16), ctx=ctx)
    W1 = nd.random_normal(loc=0, scale=weight_scale, shape=(8, 1, 1, 64), ctx=ctx )
    # b1 = nd.zeros(shape=16, ctx=ctx)
    
    # P2 = np.sqrt(6)/np.sqrt(16)
    # W2 = nd.random_uniform(low = -P2, high= P2, shape=(32, 16, 1, 8), ctx=ctx)
    W2 = nd.random_normal(loc=0, scale=weight_scale, shape=(8, 8, 1, 32), ctx=ctx )
    # b2 = nd.zeros(shape=32, ctx=ctx)
    
    # P3 = np.sqrt(6)/np.sqrt(32)
    # W3 = nd.random_uniform(low = -P3, high= P3, shape=(64, 32, 1, 8), ctx=ctx)
    W3 = nd.random_normal(loc=0, scale=weight_scale, shape=(16, 8, 1, 32), ctx=ctx )
    # b3 = nd.zeros(shape=64, ctx=ctx)
    
    # P3 = np.sqrt(6)/np.sqrt(32)
    # W3 = nd.random_uniform(low = -P3, high= P3, shape=(64, 32, 1, 8), ctx=ctx)
    W4 = nd.random_normal(loc=0, scale=weight_scale, shape=(16, 16, 1, 16), ctx=ctx )
    W5 = nd.random_normal(loc=0, scale=weight_scale, shape=(32, 16, 1, 16), ctx=ctx )
    W6 = nd.random_normal(loc=0, scale=weight_scale, shape=(32, 32, 1, 16), ctx=ctx )    
    # b3 = nd.zeros(shape=64, ctx=ctx)
    
    # P4 = np.sqrt(6)/np.sqrt(64)
    # W4 = nd.random_uniform(low = -P4, high= P4, shape=(64256, num_fc), ctx=ctx)
    W7 = nd.random_normal(loc=0, scale=weight_scale, shape=(sl, num_fc1), ctx=ctx )
#     print(W4)
    # b4 = nd.zeros(shape=num_fc, ctx=ctx) 64960  64832  64704
    W8 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc1, num_fc2), ctx=ctx )    
    
    # P5 = np.sqrt(6)/np.sqrt(64)
    # W5 = nd.random_uniform(low = -P5, high= P5, shape=(num_fc, num_outputs), ctx=ctx)
    W9 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc2, num_outputs), ctx=ctx )
    # b5 = nd.zeros(shape=num_outputs, ctx=ctx)
    
    # W1 = nd.random_normal(shape=(16, 1, 1, 16), scale=weight_scale, ctx=ctx)
    b1 = nd.random_normal(shape=8, scale=weight_scale, ctx=ctx)
    # b1 = nd.random_normal(shape=16, scale=P1, ctx=ctx)

#     W2 = nd.random_normal(shape=(32, 16, 1, 8), scale=weight_scale, ctx=ctx)
    b2 = nd.random_normal(shape=8, scale=weight_scale, ctx=ctx)
    # b2 = nd.random_normal(shape=32, scale=P2, ctx=ctx)
    
#     W3 = nd.random_normal(shape=(64, 32, 1, 8), scale=weight_scale, ctx=ctx)
    b3 = nd.random_normal(shape=16, scale=weight_scale, ctx=ctx)
    b4 = nd.random_normal(shape=16, scale=weight_scale, ctx=ctx)    
    b5 = nd.random_normal(shape=32, scale=weight_scale, ctx=ctx)    
    b6 = nd.random_normal(shape=32, scale=weight_scale, ctx=ctx)        
    # b3 = nd.random_normal(shape=64, scale=P3, ctx=ctx)
#     #  29056
#     W4 = nd.random_normal(shape=(64256, num_fc), scale=weight_scale, ctx=ctx)
    b7 = nd.random_normal(shape=num_fc1, scale=weight_scale, ctx=ctx)
    b8 = nd.random_normal(shape=num_fc2, scale=weight_scale, ctx=ctx)    
    # b4 = nd.random_normal(shape=num_fc, scale=P4, ctx=ctx)

#     W5 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
    b9 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)
    # b5 = nd.random_normal(shape=num_outputs, scale=P5, ctx=ctx)

    params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9]

    vs = []
    sqrs = []    
    
    # And assign space for gradients
    for param in params:
        param.attach_grad()
        vs.append(param.zeros_like())
        sqrs.append(param.zeros_like())        
    return params, vs, sqrs


# CNN model
def net_PRL(X, params, debug=False, pool_type='max',pool_size = 4,pool_stride=2):
    [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9] = params
    drop_prob = 0.5
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(1,64), num_filter=8, stride=(1,1),dilate=(1,1))
    h1 = nd.LeakyReLU(h1_conv, act_type='elu')
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(1,32), num_filter=8, stride=(1,1),dilate=(1,1))
    h2_pooling = nd.Pooling(data=h2_conv, pool_type=pool_type, kernel=(1,8), stride=(1,pool_stride))
    h2 = nd.LeakyReLU(h2_pooling, act_type='elu')
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))
        
    ########################
    #  Define the computation of the third convolutional layer
    ########################
    h3_conv = nd.Convolution(data=h2, weight=W3, bias=b3, kernel=(1,32), num_filter=16, stride=(1,1),dilate=(1,1))
    h3 = nd.LeakyReLU(h3_conv, act_type='elu')
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))
        
    ########################
    #  Define the computation of the 4th convolutional layer
    ########################
    h4_conv = nd.Convolution(data=h3, weight=W4, bias=b4, kernel=(1,16), num_filter=16, stride=(1,1),dilate=(1,1))
    h4_pooling = nd.Pooling(data=h4_conv, pool_type=pool_type, kernel=(1,6), stride=(1,pool_stride))
    h4 = nd.LeakyReLU(h4_pooling, act_type='elu')
    if debug:
        print("h4 shape: %s" % (np.array(h4.shape)))
        
    ########################
    #  Define the computation of the 5th convolutional layer
    ########################
    h5_conv = nd.Convolution(data=h4, weight=W5, bias=b5, kernel=(1,16), num_filter=32, stride=(1,1),dilate=(1,1))
    h5 = nd.LeakyReLU(h5_conv, act_type='elu')
    if debug:
        print("h5 shape: %s" % (np.array(h5.shape)))
        
    ########################
    #  Define the computation of the 6th convolutional layer
    ########################
    h6_conv = nd.Convolution(data=h5, weight=W6, bias=b6, kernel=(1,16), num_filter=32, stride=(1,1),dilate=(1,1))
    h6_pooling = nd.Pooling(data=h6_conv, pool_type=pool_type, kernel=(1,4), stride=(1,pool_stride))
    h6 = nd.LeakyReLU(h6_pooling, act_type='elu')
    if debug:
        print("h6 shape: %s" % (np.array(h6.shape)))        

    ########################
    #  Flattening h6 so that we can feed it into a fully-connected layer
    ########################
    h7 = nd.flatten(h6)
    if debug:
        print("Flat h7 shape: %s" % (np.array(h7.shape)))

    ########################
    #  Define the computation of the 8th (fully-connected) layer
    ########################
    h8_linear = nd.dot(h7, W7) + b7
    h8 = nd.LeakyReLU(h8_linear, act_type='elu')
    if autograd.is_training():
        # 对激活函数的输出使用droupout
        h8 = dropout(h8, drop_prob)    
    if debug:
        print("h8 shape: %s" % (np.array(h8.shape)))

    ########################
    #  Define the computation of the 9th (fully-connected) layer
    ########################
    h9_linear = nd.dot(h8, W8) + b8
    h9 = nd.LeakyReLU(h9_linear, act_type='elu')
    if autograd.is_training():
        # 对激活函数的输出使用droupout
        h9 = dropout(h9, drop_prob)    
    if debug:
        print("h9 shape: %s" % (np.array(h9.shape)))

    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h9, W9) + b9
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    
    interlayer = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9]
    
    return yhat_linear, interlayer

# Non-linear function.
def relu(X):
    return nd.maximum(X,nd.zeros_like(X))

# Loss function.
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

def transform_softmax(x):
    max_of_dim1 =nd.max(x,axis=1,keepdims=True)
    return (nd.exp(x-max_of_dim1).T/nd.exp(x-max_of_dim1).sum(axis=1,keepdims=True).T).T

def softmax_cross_entropy(yhat_linear, y):   # 交叉熵损失
    # return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
    return - nd.nansum(y * nd.log(transform_softmax(yhat_linear)), axis=0, exclude=True)

def evaluate_accuracy(data_iterator, num_examples, batch_size, params, net, pool_type,pool_size,pool_stride):
    numerator = 0.
    denominator = 0.
    for batch_i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((batch_size,1,1,-1))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output, _ = net_PRL(data, params,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
        print('Evaluating accuracy. (complete percent: %.2f/100' %(1.0 * batch_i / (num_examples//batch_size) * 100) +')' , end='')
        sys.stdout.write("\r")
    return (numerator / denominator).asscalar()

# Mini-batch stochastic gradient descent.
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
        
# Adam.
def adam(params, vs, sqrs, lr, batch_size, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    for param, v, sqr in zip(params, vs, sqrs):
        g = param.grad / batch_size

        v[:] = beta1 * v + (1. - beta1) * g
        sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

        v_bias_corr = v / (1. - beta1 ** t)
        sqr_bias_corr = sqr / (1. - beta2 ** t)

        div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
        param[:] = param - div

def Train(train, test, Debug, batch_size, lr
          , smoothing_constant, num_fc1, num_fc2, num_outputs, epochs, SNR
          , sl, pool_type ,pool_size ,pool_stride, params_init=None, period=None):
    
    num_examples = train.shape[0]
    # 训练集数据类型转换
    y = nd.array(~train.sigma.isnull() +0)
    X = nd.array(Normolise(train.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    print('Label for training:', y.shape)
    print('Dataset for training:', X.shape, end='\n\n')

    dataset_train = gluon.data.ArrayDataset(X, y)
    train_data = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True, last_batch='discard')

    y = nd.array(~test.sigma.isnull() +0)
    X = nd.array(Normolise(test.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    print('Label for testing:', y.shape)
    print('Dataset for testing:', X.shape, end='\n\n')
    
    # 这里使用data模块来读取数据。创建测试数据。  (不shuffle)
    dataset_test = gluon.data.ArrayDataset(X, y)
    test_data = gluon.data.DataLoader(dataset_test, batch_size, shuffle=True, last_batch='discard')

    
    # Train
    loss_history = []
    loss_v_history = []
    moving_loss_history = []
    test_accuracy_history = []
    train_accuracy_history = []
    
#     assert period >= batch_size and period % batch_size == 0
    
    # Initializate parameters
    if params_init:
        print('Loading params...')
        params = params_init

#         [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9] = params

        # random fc layers
        weight_scale = .01
#         W7 = nd.random_normal(loc=0, scale=weight_scale, shape=(sl, num_fc1), ctx=ctx )
#         W8 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc1, num_fc2), ctx=ctx )        
#         W9 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc2, num_outputs), ctx=ctx )
#         b7 = nd.random_normal(shape=num_fc1, scale=weight_scale, ctx=ctx)
#         b8 = nd.random_normal(shape=num_fc2, scale=weight_scale, ctx=ctx)    
#         b9 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)  

#         params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9]
        print('Random the FC1&2-layers...')

        vs = []
        sqrs = [] 
        for param in params:
            param.attach_grad()
            vs.append(param.zeros_like())
            sqrs.append(param.zeros_like())              
    else:
        params, vs, sqrs = init_params(num_fc1 = 64, num_fc2 = 64, num_outputs = 2, sl=sl)
        print('Initiate weights from random...')
        
    # Debug
    if Debug:
        print('Debuging...')
        if params_init:
            params = params_init
        else:
            params, vs, sqrs = init_params(num_fc1 = 64, num_fc2 = 64, num_outputs = 2, sl=sl)
        for data, _ in train_data:
            data = data.as_in_context(ctx).reshape((batch_size,1,1,-1))
            break
        _, _ = net_PRL(data, params, debug=Debug, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        print()
    
#     total_loss = [Total_loss(train_data_10, params, batch_size, num_outputs)]
    
    t = 0
#   Epoch starts from 1.
    print('pool_type: ', pool_type)
#     print('pool_size: ', pool_size)
    print('pool_stride: ', pool_stride)
    print('sl: ', sl)
    best_test_acc = 0
    best_params_epoch = 0

    for epoch in range(1, epochs + 1):
        Epoch_loss = []
#         学习率自我衰减。
        if epoch > 2:
#             lr *= 0.1
            lr /= (1+0.01*epoch)
        for batch_i, ((data, label),(data_v, label_v)) in enumerate(zip(train_data, test_data)):
            data = data.as_in_context(ctx).reshape((batch_size,1,1,-1))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, num_outputs)
            with autograd.record():
                output, _ = net_PRL(data, params, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
#             print(output)
#             sgd(params, lr, batch_size)

#           Increment t before invoking adam.
            t += 1
            adam(params, vs, sqrs, lr, batch_size, t)

            data_v = data_v.as_in_context(ctx).reshape((batch_size,1,1,-1))
            label_v = label_v.as_in_context(ctx)
            label_v_one_hot = nd.one_hot(label_v, num_outputs)
            output_v, _ = net_PRL(data_v, params, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
            loss_v = softmax_cross_entropy(output_v, label_v_one_hot)            
            
#             #########################
#              Keep a moving average of the losses
#             #########################
            curr_loss = nd.mean(loss).asscalar()
            curr_loss_v = nd.mean(loss_v).asscalar()
            moving_loss = (curr_loss if ((batch_i == 0) and (epoch-1 == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            loss_history.append(curr_loss)
            loss_v_history.append(curr_loss_v)
            moving_loss_history.append(moving_loss)
            Epoch_loss.append(curr_loss)
#             if batch_i * batch_size % period == 0:
#                 print('Curr_loss: ', curr_loss)

            # print('Working on epoch %d. Curr_loss: %.5f (complete percent: %.2f/100' %(epoch, curr_loss*1.0, 1.0 * batch_i / (num_examples//batch_size) * 100) +')' , end='')
            # sys.stdout.write("\r")
            # print('{"metric": "Training Loss for ALL", "value": %.5f}' %(curr_loss*1.0) )
            # print('{"metric": "Testing Loss for ALL", "value": %.5f}' %(curr_loss_v*1.0) )
            print('{"metric": "Training Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss*1.0) )
            print('{"metric": "Testing Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss_v*1.0) )
        test_accuracy = evaluate_accuracy(test_data, num_examples, batch_size, params, net_PRL,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        train_accuracy = evaluate_accuracy(train_data, num_examples, batch_size, params, net_PRL,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        test_accuracy_history.append(test_accuracy)
        train_accuracy_history.append(train_accuracy)

        if test_accuracy >= best_test_acc:
            best_test_acc = test_accuracy
            best_params_epoch = epoch


        # print("Epoch %d, Moving_loss: %.6f, Epoch_loss(mean): %.6f, Train_acc %.4f, Test_acc %.4f" %
              # (epoch, moving_loss, np.mean(Epoch_loss), train_accuracy, test_accuracy))
        print('{"metric": "Train_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), train_accuracy) )
        print('{"metric": "Test_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), test_accuracy) )
        yield (params, loss_history, loss_v_history, moving_loss_history, test_accuracy_history, train_accuracy_history, best_params_epoch)




def mkdir_checkdir(path = "/output"):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('MKDIR: ' + path + ' successful!')
    else:
        print(path + " have existed!")





import urllib
url  = 'https://dcc.ligo.org/public/0002/T0900288/003/ZERO_DET_high_P.txt'
raw_data=urllib.request.urlopen(url)
ZERO_DET = np.loadtxt(raw_data)

def noise_psd(noise_sample, lensample, fs,low_pass = 20):
    fs = 8192
    NFFT = fs//8
    NOVL = NFFT/2
    noise_sample = np.array(noise_sample)
    psd_window = np.blackman(NFFT)
    data_freqs = np.fft.fftfreq(lensample) * fs
    power_vec_, freqs_ = mlab.psd(noise_sample, Fs = fs, NFFT = NFFT, window=psd_window
                                  , noverlap=NOVL ,sides='onesided')
    slc = (freqs_>low_pass) #& (freqs_<high_pass)
    return np.interp(np.abs(data_freqs), freqs_[slc], power_vec_[slc])

def noise_psd_zero(ZERO_DET, lensample,fs = 8192,low_pass = 20):
    data_freqs = np.fft.fftfreq(lensample) * fs
    slc = (ZERO_DET[:,0]>=low_pass)# & (ZERO_DET[:,0]<=high_pass)
    asd_zero = np.interp(np.abs(data_freqs)
                         , ZERO_DET[:,0][slc]
                         , ZERO_DET[:,1][slc])
    return asd_zero**2

def SNR_MF(data, noise_sample, signal, GW_train_shape, own_noise=1, fs=8192, low_pass=20):
    lensample = data.shape[1]
    try: # Tukey window preferred, but requires recent scipy version 
        dwindow = scipy.signal.get_window(('tukey',1./8),lensample)
    except: # Blackman window OK if Tukey is not available
        dwindow = scipy.signal.get_window('blackman',lensample) 
        print('No tukey windowing, using blackman!')    
    # FFT
    data_freqs = np.fft.fftfreq(lensample) * fs
    FFT_data = np.fft.fft(data*dwindow) /fs
    FFT_signal = np.fft.fft(signal*dwindow) /fs
    
    SNR_mf = np.array([])
    for i in range(GW_train_shape):
        # PSD of noise
        if own_noise == 1: power_vec = noise_psd(noise_sample[i,:], lensample, fs,low_pass = low_pass)
        elif own_noise == 0: power_vec = noise_psd_zero(ZERO_DET, lensample,fs = 8192,low_pass = 20)
        optimal = FFT_data[i,:] * FFT_signal[i,:].conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal) * fs

        # -- Normalize the matched filter output
        df = np.abs(data_freqs[1] - data_freqs[0]) # also df=nsample/fs
        sigmasq = 1*(FFT_signal[i,:] * FFT_signal[i,:].conjugate() / power_vec).sum() * df
        sigma0 = np.sqrt(np.abs(sigmasq))
        SNR_complex = (optimal_time) / (sigma0)
        SNR_mf = np.append(SNR_mf, np.max(np.abs(SNR_complex)))
    return SNR_mf






def Normolise(data):
    data_array = np.array(data)
    data_array_shape = data_array.shape[0]
    return pd.DataFrame((data_array -np.mean(data_array, axis=1).reshape(data_array_shape,-1))/np.std(data_array, axis=1).reshape(data_array_shape,-1)
                        ,index = data.index)


def pos_gap(samples):
    positions = []
    gaps = []
    for sam in samples.values.tolist():
        position = [index for index, value in enumerate(sam) if (sam[index-1] * sam[index]  < 0) & (index != 0)]
        gaps.append([position[i+1] - j for i,j in enumerate(position) if j != position[-1] ])
        positions.append(position)
    return positions, gaps



def creat_data(GW_train, noise1, SNR):
    # GW_train = Normolise(GW_train)
    noise1array = np.array(noise1)
    GW_train_shape = GW_train.shape[0]
    GW_train_index = GW_train.index
    positions, gaps = pos_gap(Normolise(GW_train))
    max_peak = GW_train.max(axis=1)
    
    sigma = GW_train.max(axis=1) / float(SNR) / noise1array[:GW_train_shape,:].std(axis=1)
    # data = GW_train + np.multiply(noise1array[:GW_train_shape,:], sigma.reshape((GW_train_shape,-1)) )
    signal = GW_train.div(sigma, axis=0)
    data = signal + noise1array[:GW_train_shape,:]
    SNR_mf = SNR_MF(data=data, noise_sample=noise1array[:GW_train_shape,:], signal=signal
                    ,own_noise=1,GW_train_shape=GW_train_shape
                    , fs=8192, low_pass=20)
    data['SNR_mf0'] = SNR_MF(data=data, noise_sample=noise1array[:GW_train_shape,:], signal=signal
                             ,own_noise=0,GW_train_shape=GW_train_shape
                             , fs=8192, low_pass=20)
    data['SNR_mf'] = SNR_mf

    data['mass'] = GW_train_index
    data['positions'] , data['gaps'] = positions, gaps
    data['max_peak'] = max_peak
    data['sigma'] = sigma

    i = 1
    while (i+1)*GW_train_shape <= noise1array.shape[0]:
        noise1array_p = noise1array[i*GW_train_shape:(i+1)*GW_train_shape,:]

        sigma = GW_train.max(axis=1) / float(SNR) / noise1array_p[:GW_train_shape,:].std(axis=1)
        # data_new = GW_train + np.multiply(noise1array_p[:GW_train_shape,:], sigma.reshape((GW_train_shape,-1)) )
        data_new = GW_train.div(sigma, axis=0) + noise1array_p[:GW_train_shape,:]

        data_new['mass'] = GW_train_index
        data_new['positions'] , data_new['gaps'] = positions, gaps
        data_new['max_peak'] = max_peak
        data_new['sigma'] = sigma
        data = pd.concat([data, data_new ])
        i+=1
        print('Loop! ',i-1 , end='')
#         print('{"metric": "LOOP for SNR=%s", "value": %d}' %(str(SNR), int(i-1)) )
        sys.stdout.write("\r")
    return data





data_GW_train = pd.read_csv('../input/GW_data/GW_train_full.csv', index_col=0)
print('The shape of data_GW_train: ' , data_GW_train.shape)
data_GW_test = pd.read_csv('../input/GW_data/GW_test_full.csv', index_col=0)
print('The shape of data_GW_test: ' , data_GW_test.shape)



noise1 = pd.read_csv('../input/ligo_localnoise_9_9000_8192_1/LigoNose9_9000_8192_1.csv', index_col=0)
print('The shape of the noise1: ', noise1.shape)

noise_train = pd.read_csv('../input/ligo_localnoise_9_9000_8192_2/LigoNose9_9000_8192_3.csv', index_col=0)
print('The shape of the noise_train: ', noise_train.shape)

noise2 = pd.read_csv('../input/ligo_localnoise_9_9000_8192_1/LigoNose9_9000_8192_2.csv', index_col=0)
print('The shape of the noise2: ', noise2.shape)

noise_test = pd.read_csv('../input/ligo_localnoise_9_9000_8192_2/LigoNose9_9000_8192_4.csv', index_col=0)
print('The shape of the noise_test: ', noise_test.shape)






batch_size = 256
sampling_freq = 8192
colomns = [ str(i) for i in range(sampling_freq)] + ['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0']

train_dict = {}
test_dict = {}
params_ = None

for snr in list([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
    print()
    print('SNR = ', snr)
    train_data = nd.array(data_GW_train, ctx=mx.cpu())
    test_data = nd.array(data_GW_test, ctx=mx.cpu())
    peak_samppoint, peak_time = cal_peak_nd(train_data)
    print(peak_samppoint, peak_time)
    rand_times = 14
    train_, train_shift_list = shuffle_data_nd(train_data,peak_samppoint, peak_time, rand_times)
    rand_times = 13
    test_, train_shift_list = shuffle_data_nd(test_data,peak_samppoint, peak_time, rand_times)
    print(train_.shape)
    print(test_.shape)

    data_train = creat_data(pd.DataFrame(train_.asnumpy(), columns=colomns[:8192]), noise1, snr)
    print(data_train.shape)
    print(data_train.SNR_mf.mean())

    data_test = creat_data(pd.DataFrame(test_.asnumpy(), columns=colomns[:8192]), noise2, snr)
    print(data_test.shape)
    print(data_test.SNR_mf.mean())

    train_dict['%s' %int(snr*10)] = pd.concat([data_train, noise_train.iloc[:data_train.shape[0],:]])[colomns]
    test_dict['%s' %int(snr*10)] = pd.concat([data_test, noise_test.iloc[:data_test.shape[0],:]])[colomns]
    print(train_dict['%s' %int(snr*10)].shape)
    print(test_dict['%s' %int(snr*10)].shape)




    pool_type='max'
    pool_size = 4
    pool_stride= 2
# 64256 16
# 64832 6
# 64704 8
# 64960 4
# 65024 2
    sl = 31456
    address = 'SNR%s_PRL' %int(snr*10)
    mkdir_checkdir(path = "./%s" %address)



    print('Start Training at SNR = %s ...' %int(snr*10))
    
    Info = Train(train = train_dict['%s' %int(snr*10)]
                 ,test = test_dict['%s' %int(snr*10)], Debug=True , params_init = params_
                 , batch_size=256, lr=0.0003, epochs=30
                 , smoothing_constant = .01, SNR = snr
                 , sl=sl, pool_type=pool_type ,pool_size = pool_size, pool_stride=pool_stride
                 , num_fc1 = 64, num_fc2 = 64, num_outputs = 2, period = 256)


    # test_accuracy_history_final = [0]
    for index, info in enumerate(Info):
        (params, loss_history, loss_v_history, moving_loss_history, test_accuracy_history, train_accuracy_history, best_params_epoch) = info
        # Save
        for key, value in {'params':params
    #                        , 'loss_history': nd.array(loss_history)
    #                                  , 'loss_v_history': nd.array(loss_v_history)
    #                                  , 'moving_loss_history': nd.array(moving_loss_history)
    #                                  , 'test_accuracy_history': nd.array(test_accuracy_history)
    #                                  , 'train_accuracy_history': nd.array(train_accuracy_history)
                          }.items():

            # if train_accuracy_history[-1] == 1 and test_accuracy_history[-1] >= max(test_accuracy_history_final):
            #     test_accuracy_history_final.append(test_accuracy_history[-1])
            #     nd.save('/output/info_%s/%s' %(str(SNR), key), value)
            #     nd.save('./output/info_%s/%s' %(str(SNR), key), value)
            # else:
            #     pass
            nd.save("./%s/%s_%s" %(address,key,index+1), value)
    print('best_params_epoch:', best_params_epoch)

    params_ = nd.load('./SNR%s_PRL/params_%s'  %( int(snr*10) ,best_params_epoch))
    os.system('rm `ls ./%s/params_*|egrep -v ./%s/params_%s`' %(address, address, best_params_epoch))



# floyd run --gpu \
# --data wctttty/datasets/gw_colored8192/2:GW_data \
# --data wctttty/datasets/ligonose9_9000_8192/7:ligo_localnoise_9_9000_8192_1 \
# --data wctttty/datasets/ligonose9_9000_8192/6:ligo_localnoise_9_9000_8192_2 \
# -m "PRL_oldversion" \
# "bash setup_floydhub.sh && python run_old_PRL.py"
