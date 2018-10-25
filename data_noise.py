#! usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import fft, ifft, real
from scipy.signal import welch
import urllib

url  = 'https://dcc.ligo.org/public/0002/T0900288/003/ZERO_DET_high_P.txt'
raw_data=urllib.request.urlopen(url)
ZERO_DET = np.loadtxt(raw_data)

def GenNoise_matlab(nDataSamples = 16384, fLow = 40, fHigh = 1024, fs = 4096, filtOrdr = 100, debug = None):
    # Generate simulated GW detector noise
    # File containing target sensitivity curve (first column is frequency and
    #  second column is square root of PSD).
    targetSens = ZERO_DET##np.loadtxt(raw_data)

    if debug:
        # Plot the target sensitivity.
        plt.loglog(targetSens[:,0], targetSens[:,1])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'Strain Sensitivity ($1/\sqrt{Hz}$)')

    # Select pass band.
    # fLow = 40 #Hz
    # fHigh = 1024 #Hz
    targetSens[(targetSens[:,0] <= fLow) + (targetSens[:,0] >=fHigh), 1] = 0
    if debug:
        plt.loglog(targetSens[:,0], targetSens[:,1])
        plt.show()

    # Sampling frequency of the data to be generated (should be less than half
    # of the maximum frequency in target PSD.
    # fs = 4096 #Hz

    ## Design filter
    # B = fir2(N,F,A) designs an Nth order linear phase FIR digital filter with
    # the frequency response specified by vectors F and A and returns the
    # filter coefficients in length N+1 vector B.  The frequencies in F must be
    # given in increasing order with 0.0 < F < 1.0 and 1.0 corresponding to
    # half the sample rate.
    ##
    # FIR filter order
    # filtOrdr = 100
    ##
    # We only include frequencies up to the Nyquist frequency (half of sampling
    # frequency) when designing the filter.
    indxfCut = (targetSens[:,0] <= fs/2)
    targetSens = targetSens[indxfCut, :]

    # Add 0 frequency and corresponding PSD value
    # as per the requirement of FIR2. Similarly add Nyquist frequency.
    if targetSens[0,0] > 0:
        addZero = 1
    else:
        addZero = 0

    if targetSens[-1,0] < fs/2:
        addNyq = 1
    else:
        addNyq = 0
    ##
    if addZero:
        targetSens = np.concatenate((np.zeros((1,2)), targetSens))
    if addNyq:
        targetSens = np.concatenate((targetSens, np.array([[fs/2, 0]])))
    ##
    # Obtain filter coefficients. 
    b = fir2(filtOrdr, targetSens[:,0]/(fs/2), targetSens[:,1])
    ##
    if debug:
        # Compare target and designed quantities
        # Get the impulse response
        impDataNSamples = 2048
        impSample = int(np.floor(impDataNSamples/2))
        impVec = np.zeros((1,impDataNSamples))
        impVec[0][impSample-1] = 1
        impResp = fftfilt(b.reshape((-1,1)),impVec)
        ##
        # Get the transfer function
        designTf = fft(impResp)
        ##
        # Plot the magnitude of the filter transfer function.
        kNyq = int(np.floor(impDataNSamples/2)+1)
        posFreq = np.arange(kNyq) * (1/(impDataNSamples/fs))

        plt.plot(posFreq,abs(designTf[0][:kNyq]), label = 'Designed')
        plt.plot(targetSens[:,0],targetSens[:,1], label = 'Target')
        plt.ylabel('TF magnitude')
        plt.xlabel('Frequency (Hz)')
        plt.legend()
        plt.show()

    ## Generate noise
    # Pass a white noise sequence through the designed filter.
    # nDataSamples = 16384
    inputNoise = np.random.randn(1,nDataSamples)
    outputNoise = fftfilt(b.reshape((-1,1)),inputNoise)[0]
    if debug:
        plt.plot(np.arange((nDataSamples-1)+1)/fs, 
                outputNoise)
        plt.show()
    ##
    # Estimate PSD of simulated noise. *Note*: Scaling may be off because of
    # (a) factors involved in discrete version of Wiener-Khinchin theorem, and
    # (b) factors involved in how pwelch defines PSD. This can be easily
    # corrected by multiplying with an overall factor (exercise: work out the
    # factor!).
    if debug:
        [f, pxx] = welch(x = outputNoise,nperseg=2048, fs = fs)
        plt.plot(f, np.sqrt(pxx))

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$[PSD]^{1/2}$')
        plt.show()
    return outputNoise

def fftfilt(b, x, nfft=None):
    m = x.shape[0]
    if m == 1:
        x = x.reshape((-1,1))  # turn row into a column
    nx = x.shape[0]
    
    if min(b.shape) > 1:
        assert b.shape[1] == x.shape[1] and x.shape[1] <=1, "signal:fftfilt:InvalidDimensions"
    else:
        b = b.reshape((-1,1))   # make input a column
    nb = b.shape[0]
    
    if nfft == None:
        # figure out which nfft and L to use
        if (nb >= nx) or (nb > 2**20):  # take a single FFT in this case
            nfft = int(2**round(np.log(nb+nx-1)/np.log(2)))
            L = nx
            print(11111)
        else:
            fftflops = np.array([ 18,59,138,303,660,1441,3150,6875,14952,32373,69762,
                                 149647,319644,680105,1441974,3047619,6422736,13500637,
                                 28311786,59244791,59244791*2.09])
            n = 2**np.arange(1,22,1)
            validset = np.nonzero(n > (nb -1))[0]   # must have nfft > (nb-1)
            n = n[validset]
            fftflops = fftflops[validset]
            # minimize (number of blocks) * (number of flops per fft)
            L = n - (nb - 1)
            temp = np.ceil(nx/L) * fftflops
            dum, ind = np.min(temp), np.argmin(temp)
            nfft = int(n[ind])
            L = L[ind]
            
    else:  # nfft is given
        # Cast to enforce precision rules
        pass
        raise 'nfft is given?'
        '''
        nfft = signal.internal.sigcasttofloat(nfft,'double','fftfilt','N','allownumeric');
        if nfft < nb
            nfft = nb;
        end
        nfft = 2.^(ceil(log(nfft)/log(2))); % force this to a power of 2 for speed
        L = nfft - nb + 1;        
        '''
    # Check the input data type. Single precision is not supported.
    '''
    try
        chkinputdatatype(b,x,nfft);
    catch ME
        throwAsCaller(ME);
    end'''
    B = fft(b.T, nfft).T
    if b.size == 1:
        B = B.T     # make sure fft of B is a column (might be a row if b is scalar)
    if b.shape[1] == 1:
        B = np.repeat(B, [x.shape[1],],axis=1)    # replicate the column B 
    if x.shape[1] == 1:
        x = np.repeat(x, [b.shape[1],],axis=1)   # replicate the column x 
    y = np.zeros_like(x)

    istart = 1
    while istart <= nx:
        iend = min(istart+L-1, nx)
        if (iend - istart) == 0:
            X = x[istart] * np.ones((nfft,1))  # need to fft a scalar
        else:
            X = fft(x[istart-1:iend,:].T, nfft).T
        Y = ifft((X * B).T).T
        yend = min(nx,istart+nfft-1)
        y[istart-1:yend,:] = y[istart-1:yend,:] + Y[:(yend-istart+1),:]
        istart += L
    y = real(y)
    if (m == 1) and (y.shape[1] == 1):
        y = y.T    # turn column back into a row
    return y


def fir2(nn, ff, aa, **kwargs):
    '''http://read.pudn.com/downloads48/sourcecode/math/163974/fir2.m__.htm'''
    
    nn += 1
    ff = ff.reshape(1,-1)
    aa = aa.reshape(1,-1)
    assert len(kwargs) <= 3, 'Wrong number of input parameters!'

    npt = kwargs.setdefault('npt', None)
    lap = kwargs.setdefault('lap', None)
    wind = kwargs.setdefault('wind', None)
    
    length_kwargs = pd.Series(kwargs).count()
    
    if length_kwargs > 0:
        if length_kwargs == 1:
            if npt:
                if 2**round(np.log(npt)/np.log(2)) != npt:
                    npt = 2**round(np.log(npt)/np.log(2))
                wind = np.hamming(nn)
            else:
                wind = npt
                npt = 512
            lap = np.floor(npt/25)
        elif length_kwargs == 2:
            if npt ==1:
                if 2**round(np.log(npt)/np.log(2)) != npt:
                    npt = 2**round(np.log(npt)/np.log(2))
                if lap:
                    wind = np.hamming(nn)
                else:
                    wind = lap
                    lap = np.floor(npt/25)
            else:
                wind = npt
                npt = lap
                lap = np.floor(npt/25)
    elif length_kwargs == 0:
        if nn < 1024:
            npt = 512
        else:
            npt = 2**round(np.log(npt)/np.log(2))
        wind = np.hamming(nn)
        lap = np.floor(npt/25)

    assert nn == len(wind), print('The specified window must be the same as the filter length')

    [mf, nf] = ff.shape
    [ma, na] = aa.shape
    assert (mf == ma) and (nf == na), "You must specify the same number of frequencies and amplitudes"
    
    nbrk = np.maximum(mf, nf)  # length of the series
    if mf < nf:
        ff = ff.T
#         aa = aa.T

    assert (abs(ff[0]) <= np.spacing(1)) and (abs(ff[nbrk-1] -1) <=np.spacing(1)), 'The first frequency must be 0 and the last 1'
    
    # interpolate breakpoints onto large grid 
    H = np.zeros((1, npt+1))
    nint = nbrk-1
    df = np.diff(ff.conj().T)[0]
    assert any(np.diff(df) >= 0), "Frequencies must be non-decreasing"

    npt+= 1    # Length of [dc 1 2 ... nyquist] frequencies. 
    
    nb = 1
    H[0][0] = aa[0][0]
    
    for i in range(nint):
        
        if df[i] == 0:
            nb = int(nb - lap/2)
            ne = int(nb + lap)
        else:
            ne = int(np.floor(ff[i+1] * npt)[0])
            
        assert (nb >= 0) and (ne <= npt), "Too abrupt an amplitude change near end of frequency interval"
        
        j = np.arange(nb, ne+1)
        if nb == ne:
            inc = 0
        else:
            inc = (j-nb)/(ne - nb)

        H[0][nb-1:ne] = inc*aa[0][i+1] + (1 - inc)*aa[0][i]
        nb = ne + 1
    # Fourier time-shift. 
    dt = 0.5 * (nn -1)
    rad = -dt * 1j * np.pi * np.arange(npt) / (npt-1)
    H = H * np.exp(rad)
    H = np.concatenate((H, H[0][npt-2:0:-1].reshape(1,-1).conj()), axis=1) # Fourier transform of real series. 
    ht = real(ifft(H))    # Symmetric real series. 
    
    b = ht[0][:nn]  # Raw numerator. 
    b = b * wind.T  # Apply window. 
    a = 1           # Denominator. 
    return b

def pre_fir(targetSens = 'ZERO_DET_high_P.txt', fLow=20, fHigh=9000, fs = 8192, filtOrdr = 100):
    """
    Obtain filter coefficients for preparing to generate the colored noises. (ALL NUMPY)
    Input:
    - targetSens: str. File containing target sensitivity curve (first column is frequency and
                  second column is square root of PSD).
    - fLow: Select low pass. Default as 20 Hz.
    - fHigh: Select high pass. Default as 9000 Hz.
    - fs: sample of rate. Default as 8192 Hz
    - filterOrdr: FIR filter order
    
    Output:
    - b: filter coefficients. Default as an np.array in shape of (101,)
    """
    targetSens = np.loadtxt(targetSens)
    targetSens[(targetSens[:,0] <= fLow) + (targetSens[:,0] >=fHigh), 1] = 0

    indxfCut = (targetSens[:,0] <= fs/2)
    targetSens = targetSens[indxfCut, :]

    if targetSens[0,0] > 0:
        addZero = 1
    else:
        addZero = 0
    if targetSens[-1,0] < fs/2:
        addNyq = 1
    else:
        addNyq = 0

    if addZero:
        targetSens = np.concatenate((np.zeros((1,2)), targetSens))
    if addNyq:
        targetSens = np.concatenate((targetSens, np.array([[fs/2, 0]])))

    b = fir2(filtOrdr, targetSens[:,0]/(fs/2), targetSens[:,1])
    return b


def GenNoise_matlab_np(shape, b):
    """
    Generate noise(ONLY NUMPY).
    Pass a white noise sequence through the designed filter 'b'.
    Input:
    - shape: a tuple indicate the shape of sample noises.
    - b: filter coefficients. (see pre_fir() for details)
    Output:
    - outputNoise: ndarray with the shape of 'shape'.
    """    
    (numsamples, numsamplepoints) = shape
    inputNoise = np.random.randn(numsamples, numsamplepoints)
    outputNoise = fftfilt(b.reshape((-1,1)),inputNoise.T).T
    return outputNoise


def GenNoise_matlab_nd(shape, b):
    """
    Generate noise(ONLY MXNet GPU).
    Pass a white noise sequence through the designed filter 'b'.
    Input:
    - shape: a tuple indicate the shape of sample noises.
    - b: filter coefficients (in mx.ndarray). (see pre_fir() for details)
    Output:
    - outputNoise: mx.ndarray with the shape of 'shape'.
    """
    (numsamples, numsamplepoints) = shape
    inputNoise = np.random.randn(numsamples, numsamplepoints)
    outputNoise = fftfilt_nd(b.reshape((-1,1)),inputNoise.T).T
    return outputNoise

def fftfilt_nd(b, x, nfft=None):
    m = x.shape[0]
    if m == 1:
        x = x.reshape((-1,1))  # turn row into a column
    nx = x.shape[0]
    
    if min(b.shape) > 1:
        assert b.shape[1] == x.shape[1] and x.shape[1] <=1, "signal:fftfilt:InvalidDimensions"
    else:
        b = b.reshape((-1,1))   # make input a column
    nb = b.shape[0]
    
    if nfft == None:
        # figure out which nfft and L to use
        if (nb >= nx) or (nb > 2**20):  # take a single FFT in this case
            nfft = int(2**round(np.log(nb+nx-1)/np.log(2)))
            L = nx
        else:
            fftflops = nd.array([ 18,59,138,303,660,1441,3150,6875,14952,32373,69762,
                                149647,319644,680105,1441974,3047619,6422736,13500637,
                                28311786,59244791,59244791*2.09])
            n = 2**nd.arange(1,22,1)
            validset_first = nd.argmax(n>nb-1,axis=0).asscalar()
            n = nd.slice(n, begin=[int(validset_first),], end=(None,))
            fftflops = nd.slice(fftflops, begin=[int(validset_first),], end=(None,))
            # minimize (number of blocks) * (number of flops per fft)
            L = n - (nb - 1)
            temp = nd.ceil(nx/L) * fftflops
            dum, ind = nd.min(temp), nd.argmin(temp, axis=0)
            nfft = int(n[int(ind.asscalar())].asscalar())
            L = int(L[int(ind.asscalar())].asscalar())
            
    else:  # nfft is given
        # Cast to enforce precision rules
        pass
        raise 'nfft is given?'
        '''
        nfft = signal.internal.sigcasttofloat(nfft,'double','fftfilt','N','allownumeric');
        if nfft < nb
            nfft = nb;
        end
        nfft = 2.^(ceil(log(nfft)/log(2))); % force this to a power of 2 for speed
        L = nfft - nb + 1;        
        '''
    # Check the input data type. Single precision is not supported.
    '''
    try
        chkinputdatatype(b,x,nfft);
    catch ME
        throwAsCaller(ME);
    end'''
    B = fft(b.T, nfft).T
    if b.size == 1:
        B = B.T     # make sure fft of B is a column (might be a row if b is scalar)
    if b.shape[1] == 1:
        B = np.repeat(B, [x.shape[1],],axis=1)    # replicate the column B 
    if x.shape[1] == 1:
        x = np.repeat(x, [b.shape[1],],axis=1)   # replicate the column x 
    y = np.zeros_like(x)

    istart = 1
    while istart <= nx:
        iend = min(istart+L-1, nx)
        if (iend - istart) == 0:
            X = x[istart] * np.ones((nfft,1))  # need to fft a scalar
        else:
            X = fft(x[istart-1:iend,:].T, nfft).T
        Y = ifft((X * B).T).T
        yend = min(nx,istart+nfft-1)
        y[istart-1:yend,:] = y[istart-1:yend,:] + Y[:(yend-istart+1),:]
        istart += L
    y = real(y)
    if (m == 1) and (y.shape[1] == 1):
        y = y.T    # turn column back into a row
    return y
    