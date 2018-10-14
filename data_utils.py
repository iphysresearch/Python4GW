#! usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function

from utils import *
ctx = check_ctx()
# ctx = mx.gpu()

# Data manipulation
import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import pandas as pd
from functools import reduce

# Signal processing
from scipy import signal
import scipy
import matplotlib.mlab as mlab
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.fftpack import ifft

import urllib
import sys

url  = 'https://dcc.ligo.org/public/0002/T0900288/003/ZERO_DET_high_P.txt'
raw_data = urllib.request.urlopen(url)
ZERO_DET = np.loadtxt(raw_data)
# ZERO_DET = np.loadtxt('ZERO_DET_high_P.txt')

##########################################################################################
# 质量参数对生成器
# Distribution_of_masses
##########################################################################################
def Distribution_of_masses(**kwds):
    """
    输出双黑洞质量的分布。(ratio_step 不能小于 0.01，可以通过增大 doa=100 提高精度)
    
    Input:
    共有三种输入方式，如下面的例子：
    Eg1: mass1_scope = (5,75), mass2_scope = (5,75), mass_step = 2
    Eg2: mass1_scope = (5,75), mass_step = 2, ratio_scope = (0.1,1), ratio_step = 0.1
    Eg3: Mass_scope = (5, 300), Mass_step = 1, ratio_scope = (0.01, 1), ratio_step = 0.05
    
    Output:
    A list of tuples with masses in it.
    """
    doa = 100 # ratio 的精度设计
    if 'mass1_scope' in kwds and 'mass2_scope' in kwds and 'mass_step' in kwds and 'ratio_scope' not in kwds:
        (m1s, m1e), (m2s, m2e), m_step = kwds['mass1_scope'], kwds['mass2_scope'], kwds['mass_step']
        return sorted([(m1, m2) for m1 in range(m1s, m1e +m_step, m_step) for m2 in range(m2s, m2e +m_step, m_step) if m1 >= m2])
    
    elif 'mass1_scope' in kwds and 'mass_step' in kwds and 'ratio_scope' in kwds and 'ratio_step' in kwds and 'Mass_scope' not in kwds:
        (m1s, m1e), m_step = kwds['mass1_scope'], kwds['mass_step']
        (rs, re), r_step = kwds['ratio_scope'], kwds['ratio_step']
        return sorted([(m1, m1*ratio/doa) for m1 in range(m1s, m1e +m_step, m_step) for ratio in range(int(rs*doa), int(re*doa+r_step*doa), int(r_step*doa)) if m1 + m1*ratio/doa <= m1e ])

    elif 'Mass_scope' in kwds and 'Mass_step' in kwds and 'ratio_scope' in kwds and 'ratio_step' in kwds and 'mass1_scope' not in kwds:
        (Ms, Me), M_step = kwds['Mass_scope'], kwds['Mass_step']
        (rs, re), r_step = kwds['ratio_scope'], kwds['ratio_step']
        return sorted([(doa*M/(doa+ratio), ratio*M/(doa+ratio)) for M in range(Ms, Me, M_step) for ratio in range(int(rs*doa), int(re*doa+r_step*doa), int(r_step*doa)) if doa*M/(doa+ratio) >= ratio*M/(doa+ratio)])
    
    else:
        raise KeyError("Something wrong on the keywords!")
# 
# 
# 
# 
#        
##########################################################################################
# 自制 单边 PSD 函数
# oneSidedPeriodogram_nd  -----  only for GPU (with mxnet)
# oneSidedPeriodogram     -----  using numpy
##########################################################################################
def oneSidedPeriodogram_nd(y, fs):
    """
    单边 PSD (only for GPU in mxnet.ndarray)
    
    Input:
    - y: complex ndarray. A signal
    - fs: sampling rate. Int.
    
    Output:
    - yf: Fourier series (no use)
    - xf[xf>=0]: Discrete Fourier Transform sample frequencies (one-sided)
    - oneSidedPeriodogram: one-sided PSD
    """
    N = y.shape[1]
    yf = mx.contrib.ndarray.fft(y)
    xf = fftfreq(N, 1./fs)
    oneSidedPeriodogram = 2/fs*nd.sqrt(yf[:,:N][:,::2]**2 + yf[:,:N][:,1::2]**2)**2/N
    return yf, xf[xf>=0], oneSidedPeriodogram


def oneSidedPeriodogram(y, fs, scipy = True):
    """
    单边 PSD
    
    Input:
    - y: complex ndarray. A signal
    - fs: sampling rate. Int.
    - scipy: True by dauflt.
    
    Output:
    - yf: Fourier series (no use)
    - xf[xf>=0]: Discrete Fourier Transform sample frequencies (one-sided)
    - oneSidedPeriodogram: one-sided PSD
    """
    N = y.shape[1]
    if scipy: yf = fft(y)
    else: yf = np.fft.fft(y)
    
    if scipy: xf = fftfreq(N, 1./fs)
    else: xf = np.fft.fftfreq(N, 1./fs)
    
    oneSidedPeriodogram = 2/fs*abs(yf[:,xf>=0])**2/N
    return yf, xf[xf>=0], oneSidedPeriodogram
# 
# 
# 
# 
# 
##########################################################################################
# 根据给定 ASD 生成噪声函数
# Pre_zero   前置函数，用来对 Zero_DET 模板数据进行插值和取定频谱范围
# TimeseriesFromPSD      -----  using numpy
# TimeseriesFromPSD_nd   -----  only for GPU (with mxnet)
##########################################################################################
def Pre_zero(ZERO_DET = ZERO_DET, size = (2,8192), fs = 8192, fmin = 20, fmax = 4000):
    (D, *N) = size  # N is a list
    low_f_max = fmin
    high_f_min = fmax
    # Interpolation
    freqs = fftfreq(N[-1], 1./fs)
    asd_zero = np.interp(freqs[(freqs>=ZERO_DET[:,0].min())&(freqs<=high_f_min)], ZERO_DET[:,0], ZERO_DET[:,1]) 

    shiftsize = int(low_f_max - ZERO_DET[:,0].min())
    xf = fftfreq(N[-1], 1./fs)
    xf_noise = xf[xf>=0]
    slc, slc_, slc__ = (xf_noise >= low_f_max)&(xf_noise<=high_f_min), (xf_noise < low_f_max), (xf_noise > high_f_min)

    if ctx == mx.gpu():
        asd_zero = nd.array(asd_zero, ctx = ctx, dtype='float64')
        asd_pos = nd.square(asd_zero)[shiftsize * N[-1]//8192:]
        asd_neg = nd.square(asd_zero)[shiftsize * N[-1]//8192:][::-1]    
    elif ctx == mx.cpu():
        asd_pos = np.square(asd_zero)[shiftsize:]
        asd_neg = np.square(asd_zero)[shiftsize:][::-1]            
    else:
        raise

    assert slc_.argmin() == slc.argmax()
    low_f = slc_.argmin()
    high_f = slc[slc.argmax():].argmin()+slc.argmax()
    high_f_ = N[-1]//2 - slc__.argmax()
    assert asd_pos.shape[0] == high_f - low_f
#    print(asd_neg)
    return (asd_pos, asd_neg, low_f, high_f, high_f_, size, fs, fmin, fmax)
# 
# 
def TimeseriesFromPSD(param_noise):
    """
    From ZERO_DET(PSD) to noise.
    
    Input:
    - N: the number of sampling points.(default 8192)
    - fs: sampling rate.(default 8192Hz)
    - fmin: the lowest frequency.(default 20Hz)
    - fmax: the highest frequency.(default 4000Hz)
    
    Output:
    - timeseries: ndarray
    - psd_twosided: the corresponding twosided psd
    """
    (asd_pos, asd_neg, low_f, high_f, high_f_, size, fs, fmin, fmax) = param_noise
    (*D_, N) = size

    D = reduce(lambda x, y: x * y, D_)
    # Gauss noise and its one-sided PSD without window
    gauss_noise = 1* np.random.normal(loc=0,scale=64,size=(D,N))
    _, xf_noise, psd_gauss = oneSidedPeriodogram(gauss_noise, fs)
    
    # psd using matlab
    # psd_gauss ,xf_noise= plt.mlab.psd(gauss_noise, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
    # xf_noise = xf_noise[1:]
    # psd_gauss = psd_gauss[1:]

    # Two-sided PSD
    psd_twosided  = np.hstack((  # low positive
                              np.zeros((D, low_f)), 
                                # high positive
                              psd_gauss[:, low_f:high_f] * asd_pos, 
                              np.zeros((D, high_f_)),
                              np.zeros((D, high_f_)),
                                # high negative
                              psd_gauss[:, low_f:high_f][::-1] * asd_neg, 
                                # low negative
                              np.zeros((D, low_f)) ))    

    amplitude =  np.sqrt(psd_twosided *2 *fs*N )
    epsilon = np.random.rand(D, N)*1j*2*np.pi
    timeseries = np.real(ifft(amplitude*np.exp(epsilon)))
    return timeseries.reshape(size), psd_twosided
# 
# 
def TimeseriesFromPSD_nd(param_noise):
    """
    GPU only
    """
    (asd_pos, asd_neg, low_f, high_f, high_f_, size, fs, fmin, fmax) = param_noise
    (*D_, N) = size
    D = reduce(lambda x, y: x * y, D_)
    # Gauss noise and its one-sided PSD without window
    gauss_noise = 1* nd.random_normal(loc=0,scale=64,shape=(D, N), ctx=ctx)
    _, xf_noise, psd_gauss = oneSidedPeriodogram_nd(gauss_noise, fs=8192)
    psd_gauss = nd.array(psd_gauss, ctx = ctx, dtype='float64')

    psd_twosided  = nd.concat(  # low positive
                              nd.zeros((D, low_f), ctx = ctx, dtype='float64'), 
                                # high positive
                              psd_gauss[:, low_f:high_f] * asd_pos, 
                              nd.zeros((D, high_f_), ctx = ctx, dtype='float64'),
                              nd.zeros((D, high_f_), ctx = ctx, dtype='float64'),
                                # high negative
                              psd_gauss[:, low_f:high_f][::-1] * asd_neg, 
                                # low negative
                              nd.zeros((D, low_f), ctx = ctx, dtype='float64'), dim=1)
    amplitude =  nd.sqrt(psd_twosided *2 *fs*N )
    epsilon_imag = nd.random_uniform(low=0, high=1, shape=(D,N),ctx=ctx,dtype='float64')*2*np.pi
    re = nd.cos(epsilon_imag)*amplitude
    im = nd.sin(epsilon_imag)*amplitude
    temp = nd.zeros((D, N*2) , ctx=ctx)
    temp[:,::2] = re
    temp[:,1::2] = im
    timeseries = mx.contrib.ndarray.ifft(temp)/N
    return timeseries.reshape(size),  psd_twosided
# 
#
def cal_peak_nd(data):
    peak_samppoint = data.argmax(axis=1)
    peak_samppoint = int(peak_samppoint.sum().asnumpy() / peak_samppoint.shape[0])
    peak_time = peak_samppoint/data.shape[-1]
    peak_time = float('{:.2f}'.format(peak_time))
    return peak_samppoint, peak_time
#
# 
def forward_moving_wave_np(data, a):
    return np.concatenate((data[:,a:], np.ones(shape=(data.shape[0], a)) * data[0,-1]), axis = 1)

def shuffle_data_np(data, peak_samppoint, peak_time, times):
    shift_list = np.random.uniform(0, peak_samppoint - round((peak_time-0.2)*data.shape[-1]), size = (times))
    base = forward_moving_wave_np(data, int(shift_list[0]))
    
    for shift_size in shift_list[1:]:
        temp = forward_moving_wave_np(data, int(shift_size))
        base = np.concatenate((base, temp) , axis = 0)    
    return base
# 
def forward_moving_wave_nd(data, a):
    return nd.concatenate([data[:,a:], nd.ones(shape=(data.shape[0], a), ctx=ctx) * data[0,-1]], axis = 1)
#
def shuffle_data_nd(data, peak_samppoint, peak_time, times):
    shift_list = nd.random_uniform(0, peak_samppoint - round((peak_time-0.2)*data.shape[-1]), shape=(10), ctx=ctx)
    base = forward_moving_wave_nd(data, int(shift_list.asnumpy()[0]))
    
    for shift_size in shift_list[1:]:
        temp = forward_moving_wave_nd(data, int(shift_size.asnumpy()[0]))
        base = nd.concatenate([base, temp] , axis = 0)    
    return base
# 
# 
##########################################################################################
# zero mean and unit variance as it makes traning process easier
# Normolise     -----  using numpy
# Normolise_nd  -----  using mxnet.ndarray
##########################################################################################
def Normolise(data):
    """
    Zero mean and unit variance as it makes traning process easier (each row).
    
    Input:
    - data: List, Array or DataFrame (prefered).
    
    Return:
    - DataFrame.
    """
    data_array = np.array(data)
    data_array_shape = data_array.shape[0]
    data_norm = (data_array -np.mean(data_array, axis=1).reshape(data_array_shape,-1))
    data_norm /= np.std(data_array, axis=1).reshape(data_array_shape,-1)
    try:
        return pd.DataFrame(data_norm, index = data.index)
    except:
        return data_norm
#
#
def Normolise_nd(X, num_channel):
    """
    Zero mean and unit variance as it makes traning process easier (each row).
    
    Input:
    - data: List, Array or DataFrame (prefered).
    
    Return:
    - DataFrame.
    """
    mean = X.mean(axis=2).reshape((-1,num_channel,1))
    var = nd.sqrt(((X - mean) ** 2).mean(axis=2)).reshape((-1,num_channel,1))

    data_norm = (X -mean)
    data_norm /= var
    return data_norm    
#
# 
# 
# 
# 
# 下面是早期直接生成数据集的代码
##########################################################################################
# -- To calculate the PSD of the data, choose an overlap and a window (common to all detectors)
#   that minimizes "spectral leakage" https://en.wikipedia.org/wiki/Spectral_leakage
##########################################################################################
def noise_psd(noise_sample, fs=8192, low_pass = 20):
    """
    Evalate the one-sided PSD of a tiem-series. (Using mlab.psd).
    By default, we consider the blackman window in 1/8 sampling rate
    with a 50% overlap and low-pass 20Hz.
    
    REF: https://losc.ligo.org/s/events/GW150914/LOSC_Event_tutorial_GW150914.html#Matched-filtering-to-find-the-signal
    
    Input:
    - noise_sample: List, Array or DataFrame (prefered). 
    - fs: Default fs=8192Hz. Sampling rate.
    - low_pass: Default low_pass = 20Hz.
    
    Return:
    - One-sided PSD: Array
    """
    NFFT = fs//8
    NOVL = NFFT/2
    noise_sample = np.array(noise_sample)
    lensample = noise_sample.shape[-1]
    psd_window = np.blackman(NFFT)
    data_freqs = np.fft.fftfreq(lensample) * fs
    power_vec_, freqs_ = mlab.psd(noise_sample, Fs = fs, NFFT = NFFT, window=psd_window
                                  , noverlap=NOVL ,sides='onesided')
    slc = (freqs_>low_pass) #& (freqs_<high_pass)
    return np.interp(np.abs(data_freqs), freqs_[slc], power_vec_[slc])



def noise_psd_zero(ZERO_DET, lensample,fs = 8192,low_pass = 20):
    """
    Evalate PSD (not work well).
    Please refer to noise_psd().
    """
    data_freqs = np.fft.fftfreq(lensample) * fs
    slc = (ZERO_DET[:,0]>=low_pass)# & (ZERO_DET[:,0]<=high_pass)
    asd_zero = np.interp(np.abs(data_freqs)
                         , ZERO_DET[:,0][slc]
                         , ZERO_DET[:,1][slc])
    return asd_zero**2



def SNR_MF(data, noise_sample, signal, own_noise=1, fs=8192, low_pass=20):
    """
    Evaluate optimal Mached-filtered SNR.
    
    REF: https://losc.ligo.org/s/events/GW150914/LOSC_Event_tutorial_GW150914.html#Matched-filtering-to-find-the-signal
    
    Input:
    - data: Array or DataFrame. A dataset of signals mixed with noise.
    - noise_sample: Array or DataFrame. A dataset for pure colored noise.
    - signal: Array or DataFrame. A dataset for puse GW waveform.
    - own_noise: Default 1 corresponding to the def. of 'noise_psd' PSD evaluation (Using mlab.psd).
    - fs: Default fs=8192Hz. Sampling rate.
    - low_pass: Default low_pass = 20Hz.
    
    Return:
    - SNR_mf: An array with the shape (signal.shape[0], 1)
    """
    lensample = data.shape[1]
    GW_train_shape = signal.shape[0]

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
        if own_noise == 1: power_vec = noise_psd(noise_sample[i,:], fs, low_pass)
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



def pos_gap(samples):
    """
    Calculate the positons and gaps of GW waveform in sample points during each half of the period of fluctuations.
    
    Input:
    - samples: Dataframe.
    
    Return:
    - positions: List.
    - gaps: List
    
    Example:
    >>> positions, gaps = pos_gap(Normolise(GW_train))
    """
    positions = []
    gaps = []
    for sam in samples.values.tolist():
        position = [index for index, value in enumerate(sam) if (sam[index-1] * sam[index]  < 0) & (index != 0)]
        gaps.append([position[i+1] - j for i,j in enumerate(position) if j != position[-1] ])
        positions.append(position)
    return positions, gaps



def creat_data(GW_train, noise1, SNR):
    """
    Combine the two data blocks of GW waveform and pure colored noise.
    
    Input:
    - GW_train: A DataFram with pure GW waveform.
    - noise1: An array or DataFram with pure colored noise.
    - SNR: A scalar corrsponding to a rate of the maximum of GW waveform and standard varience of noise.
    
    Return:
    - data: A dataset with mixed GW signals.
    """
    noise1array = np.array(noise1)
    GW_train_shape = GW_train.shape[0]
    GW_train_index = GW_train.index
    positions, gaps = pos_gap(Normolise(GW_train))
    max_peak = GW_train.max(axis=1)
    
    sigma = GW_train.max(axis=1) / float(SNR) / noise1array[:GW_train_shape,:].std(axis=1)
    signal = GW_train.div(sigma, axis=0)
    data = signal + noise1array[:GW_train_shape,:]
    SNR_mf = SNR_MF(data=data, noise_sample=noise1array[:GW_train_shape,:], signal=signal
                    ,own_noise=1 , fs=8192, low_pass=20)
    data['SNR_mf0'] = SNR_MF(data=data, noise_sample=noise1array[:GW_train_shape,:], signal=signal
                             ,own_noise=0, fs=8192, low_pass=20)
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
        
        SNR_mf = SNR_MF(data=data_new, noise_sample=noise1array_p[:GW_train_shape,:], signal=GW_train.div(sigma, axis=0)
                        ,own_noise=1 , fs=8192, low_pass=20)
        data_new['SNR_mf0'] = SNR_MF(data=data_new, noise_sample=noise1array_p[:GW_train_shape,:], signal=GW_train.div(sigma, axis=0)
                        ,own_noise=1 , fs=8192, low_pass=20)
        data_new['SNR_mf'] = SNR_mf

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
