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


def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃。
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape, ctx=ctx) < keep_prob
    return mask * X / keep_prob


# CNN model
def net(X, params, debug=False, pool_type='avg',pool_size = 16,pool_stride=2, act_type = 'relu', dilate_size = 1, nf=1):
    [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5] = params
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(1,16), num_filter=int(16*nf), stride=(1,1),dilate=(1,dilate_size))
    h1_activation = activation(h1_conv, act_type = act_type)
    h1 = nd.Pooling(data=h1_activation, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(1,8), num_filter=int(32*nf), stride=(1,1),dilate=(1,dilate_size))
    h2_activation = activation(h2_conv, act_type = act_type)
    h2 = nd.Pooling(data=h2_activation, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))
        
    ########################
    #  Define the computation of the third convolutional layer
    ########################
    h3_conv = nd.Convolution(data=h2, weight=W3, bias=b3, kernel=(1,8), num_filter=int(64*nf), stride=(1,1),dilate=(1,dilate_size))
    h3_activation = activation(h3_conv, act_type = act_type)
    h3 = nd.Pooling(data=h3_activation, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))

    ########################
    #  Flattening h3 so that we can feed it into a fully-connected layer
    ########################
    h4 = nd.flatten(h3)
    if debug:
        print("Flat h4 shape: %s" % (np.array(h4.shape)))

    ########################
    #  Define the computation of the 4th (fully-connected) layer
    ########################
    h5_linear = nd.dot(h4, W4) + b4
    h5 = activation(h5_linear, act_type = act_type)
    if autograd.is_training():
        # 对激活函数的输出使用droupout
        h5 = dropout(h5, drop_prob)
    if debug:
        print("h5 shape: %s" % (np.array(h5.shape)))
        print("Dropout: ", drop_prob)

    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h5, W5) + b5
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    
    interlayer = [h1, h2, h3, h4, h5]
    
    return yhat_linear, interlayer


# CNN model
def net_PLB(X, params, debug=False, pool_type='max',pool_size = 4,pool_stride=4):
    [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7] = params
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(1,16), num_filter=64, stride=(1,1),dilate=(1,1))
    h1_pooling = nd.Pooling(data=h1_conv, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))    
    h1 = relu(h1_pooling)
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(1,16), num_filter=128, stride=(1,1),dilate=(1,2))
    h2_pooling = nd.Pooling(data=h2_conv, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))
    h2 = relu(h2_pooling)
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))
        
    ########################
    #  Define the computation of the third convolutional layer
    ########################
    h3_conv = nd.Convolution(data=h2, weight=W3, bias=b3, kernel=(1,16), num_filter=256, stride=(1,1),dilate=(1,2))
    h3_pooling = nd.Pooling(data=h3_conv, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))
    h3 = relu(h3_pooling)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))
        
    ########################
    #  Define the computation of the 4th convolutional layer
    ########################
    h4_conv = nd.Convolution(data=h3, weight=W4, bias=b4, kernel=(1,32), num_filter=512, stride=(1,1),dilate=(1,2))
    h4_pooling = nd.Pooling(data=h4_conv, pool_type=pool_type, kernel=(1,pool_size), stride=(1,pool_stride))
    h4 = relu(h4_pooling)
    if debug:
        print("h4 shape: %s" % (np.array(h4.shape)))

    ########################
    #  Flattening h4 so that we can feed it into a fully-connected layer
    ########################
    h5 = nd.flatten(h4)
    if debug:
        print("Flat h5 shape: %s" % (np.array(h5.shape)))

    ########################
    #  Define the computation of the 5th (fully-connected) layer
    ########################
    h6_linear = nd.dot(h5, W5) + b5
    h6 = relu(h6_linear)
    if debug:
        print("h6 shape: %s" % (np.array(h6.shape)))

    ########################
    #  Define the computation of the 6th (fully-connected) layer
    ########################
    h7_linear = nd.dot(h6, W6) + b6
    h7 = relu(h7_linear)
    if debug:
        print("h7 shape: %s" % (np.array(h7.shape)))

    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h7, W7) + b7
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    
    interlayer = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7]
    
    return yhat_linear, interlayer


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

# Activation type
def activation(X, act_type = 'relu'):
    if act_type == 'relu':
        return nd.maximum(X,nd.zeros_like(X))
    elif act_type == 'elu':
        return nd.LeakyReLU(X, act_type=act_type)
    else:
        print('Something wrong with the act_type!')

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
    data_array = np.array(data).astype('float64')
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






# data_GW_train = pd.read_csv('../input/GW_data/GW_train_full.csv', index_col=0)
# print('The shape of data_GW_train: ' , data_GW_train.shape)
data_GW_test = pd.read_csv('../input/GW_data/GW_test_full.csv', index_col=0)
print('The shape of data_GW_test: ' , data_GW_test.shape)



# noise1 = pd.read_csv('../input/ligo_localnoise_9_9000_8192_1/LigoNose9_9000_8192_1.csv', index_col=0)
# print('The shape of the noise1: ', noise1.shape)

# noise_train = pd.read_csv('../input/ligo_localnoise_9_9000_8192_2/LigoNose9_9000_8192_3.csv', index_col=0)
# print('The shape of the noise_train: ', noise_train.shape)

noise2 = pd.read_csv('../input/ligo_localnoise_9_9000_8192/LigoNose9_9000_8192_2.csv', index_col=0)
print('The shape of the noise2: ', noise2.shape)

noise_test = pd.read_csv('../input/ligo_localnoise_9_9000_8192/LigoNose9_9000_8192_4.csv', index_col=0).head(-1)  # 最后一个有瑕疵~~
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
    # train_data = nd.array(data_GW_train, ctx=mx.cpu())
    test_data = nd.array(data_GW_test, ctx=mx.cpu())
    peak_samppoint, peak_time = cal_peak_nd(test_data)
    print(peak_samppoint, peak_time)
    # rand_times = 14
    # train_, train_shift_list = shuffle_data_nd(train_data,peak_samppoint, peak_time, rand_times)
    rand_times = 13
    test_, train_shift_list = shuffle_data_nd(test_data,peak_samppoint, peak_time, rand_times)
    # print(train_.shape)
    print(test_.shape)

    # data_train = creat_data(pd.DataFrame(train_.asnumpy(), columns=colomns[:8192]), noise1, snr)
    # print(data_train.shape)
    # print(data_train.SNR_mf.mean())

    data_test = creat_data(pd.DataFrame(test_.asnumpy(), columns=colomns[:8192]), noise2, snr)
    print(data_test.shape)
    print(data_test.SNR_mf.mean())

    # train_dict['%s' %int(snr*10)] = pd.concat([data_train, noise_train.iloc[:data_train.shape[0],:]])[colomns]
    test_dict['%s' %int(snr*10)] = pd.concat([data_test, noise_test.iloc[:data_test.shape[0],:]])[colomns]
    # print(train_dict['%s' %int(snr*10)].shape)
    print(test_dict['%s' %int(snr*10)].shape)



model_names = ['OURs', 'PLB', 'PRL']
models= [net, net_PLB, net_PRL]
SNR_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

for model_name, model in zip(model_names, models):


    AUC = {}

    SNR_MF_list = []
    for SNR_ in SNR_list:  # 不同的模型参数
        auc_ = []
        address = '../input/%s_oldversion/SNR%s_%s/' %(model_name, int(SNR_*10) ,model_name)
        params_ = nd.load(address+'%s' %(os.listdir(address)[0]))
        print(test_dict['%s' %int(SNR_*10)].head())
        SNR_MF_list.append(test_dict['%s' %int(SNR_*10)].dropna().SNR_mf.values.mean(axis=0))
        print('%s Model for SNR = %s' %(model_name, int(SNR_*10) ) )

        for SNR in SNR_list:  # 不同的数据集
            probas_pred_ = []
            ytest_true_ = []
            interlayer_ = []
            # 制作测试集
            y = nd.array(~test_dict['%s' %int(SNR*10)].sigma.isnull() +0)
            X = nd.array(Normolise(test_dict['%s' %int(SNR*10)].drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))

            num_examples = y.shape[0]
            if not num_examples%832:
                batch_size = 832
            else:
                print('Now! Reset the batch_size!')
                batch_size = 832
                # raise

            dataset_test = gluon.data.ArrayDataset(X, y)
            test_data = gluon.data.DataLoader(dataset_test, batch_size, shuffle=True, last_batch='keep')


            for batch_i, (data, label) in enumerate(test_data):
                data = data.as_in_context(ctx).reshape((data.shape[0],1,1,-1))
                label = label.as_in_context(ctx)
                output, _ = model(data, params_)
                
                probas_pred_.extend(transform_softmax(output)[:,1].asnumpy())  # 保存每个概率
                ytest_true_.extend(label.asnumpy().tolist())   # 保存每个真实标签结果

                print('(Dataset(SNR = %s), complete percent: %.2f/100' %(SNR,  1.0 * batch_i *batch_size/ (num_examples) * 100) +')' , end='')
                sys.stdout.write("\r")
            print()
            try:
                fpr, tpr, _ = roc_curve(ytest_true_, probas_pred_)
            except ValueError as e:
                print(e)
                break
            auc_.append(auc(fpr, tpr))
        
        AUC['%s' %int(SNR_*10)] = auc_
        print()
        print('Finished!', end='')
        sys.stdout.write("\r")
    np.save('AUC_%s_oldversion' %(model_name), AUC)





# floyd run --gpu \
# --data wctttty/datasets/gw_colored8192/2:GW_data \
# --data wctttty/datasets/ligonose9_9000_8192/9:ligo_localnoise_9_9000_8192 \
# --data wctttty/projects/python4gw/263:OURs_oldversion \
# --data wctttty/projects/python4gw/260:PRL_oldversion \
# --data wctttty/projects/python4gw/255:PLB_oldversion \
# -m "AUC_OURs/PLB/PRL_oldversion" \
# "bash setup_floydhub.sh && python AUC_OURs_PLB_PRL_oldversion.py"
