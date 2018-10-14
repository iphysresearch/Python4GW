#! usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function

# Data manipulation
import numpy as np
# import pandas as pd

# Signal processing
# from scipy import signal
# import scipy
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# import urllib
# import sys


def Plot_masses_scratch(masses, masses_=None):
    masses = np.array(masses)
    if masses_ is not None:
        masses_ = np.array(masses_)
    
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.scatter(masses[:,0], masses[:,1], s = 0.5)
    if masses_ is not None:
        plt.scatter(masses_[:,0], masses_[:,1], s = 0.5)
    plt.xlim(0, masses.max()*(1+0.01))
    plt.ylim(0, masses.max()*(1+0.01))
    plt.xlabel('m1'), plt.ylabel('m2')
    
    plt.subplot(1,2,2)
    plt.scatter(masses[:,1]/masses[:,0], masses[:,0] + masses[:,1], s = 0.5)
    if masses_ is not None:
        plt.scatter(masses_[:,1]/masses_[:,0], masses_[:,0] + masses_[:,1], s = 0.5)
    plt.xlabel('ratio of masses'), plt.ylabel('total masses')
    plt.show()
    
    
