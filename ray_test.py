#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time

### importing the library

# import pandas as pd
# import numpy as np

# import os, sys, time

ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)

@ray.remote
def Panyi_detector(i):
    # os.system('./Panyi_code/Panyi --m1 10 --m2 10 --sample-rate 8192 --f-min 20 --outname waveform%s \
    #                               --spin1x 0.5 --spin1y 0.5 --spin1z 0 \
    #                               --spin2x 0 --spin2y 0 --spin2z 0 \
    #                               --inclination 0 --distance 100' %i)
    # os.system('./detector_strain/MYdetector_strain -D H1 -a 1:23:45 -d 45.0 -p 30.0 -t 1000000000 < waveform%s >           H1_waveform%s' %(i,i))
    # os.system('./detector_strain/MYdetector_strain -D L1 -a 1:23:45 -d 45.0 -p 30.0 -t 1000000000 < waveform%s > L1_waveform%s' %(i,i))
    # # os.system('./detector_strain/MYdetector_strain -D V1 -a 1:23:45 -d 45.0 -p 30.0 -t 1000000000 < waveform > V1_waveform')

    # H1 = pd.DataFrame(np.loadtxt('./H1_waveform%s' %i))
    # L1 = pd.DataFrame(np.loadtxt('./L1_waveform%s' %i))
    # print(H1.shape, L1.shape)
    # # Plot
    # # plt.figure(figsize=(10,3))
    # # plt.subplot(1,3,1)
    # # plt.plot(H1[0][-8192:],H1[1][-8192:])
    # # plt.subplot(1,3,2)
    # # plt.plot(L1[0][-8192:],L1[1][-8192:])
    # # plt.subplot(1,3,3)
    # # plt.plot(L1[0][-8192:],L1[1][-8192:])
    # # plt.plot(H1[0][-8192:],H1[1][-8192:])
    # # plt.show()
    # # Check
    # print(np.allclose(H1[0], L1[0])) # 一致的 GPS 时间
    # print(H1[1].argmax())
    # print(L1[1].argmax())   # 不一致的 peak 抵达的 GPS 时间
    # os.system('rm waveform%s H1_waveform%s L1_waveform%s' %(i,i,i))
    pass