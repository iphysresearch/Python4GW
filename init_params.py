#! usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function
from utils import *
ctx = check_ctx()

# importing MxNet >= 1.0
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd, gluon

import random


def test_ctx():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    print('CPU or GPU? : ', ctx)


def init_params(num_fc1, num_fc2, num_outputs, sl, nf=1/2):
    #######################
    #  Set the scale for weight initialization and choose
    #  the number of hidden units in the fully-connected layer
    #######################
    weight_scale = .01

    W1 = nd.random_normal(loc=0, scale=weight_scale, shape=(int(16*nf), 1, 1, 16), ctx=ctx )
    W2 = nd.random_normal(loc=0, scale=weight_scale, shape=(int(32*nf), int(16*nf), 1, 8), ctx=ctx )
    W3 = nd.random_normal(loc=0, scale=weight_scale, shape=(int(64*nf), int(32*nf), 1, 8), ctx=ctx )
    W4 = nd.random_normal(loc=0, scale=weight_scale, shape=(sl, 64), ctx=ctx )
    W5 = nd.random_normal(loc=0, scale=weight_scale, shape=(64, 64), ctx=ctx )
    W6 = nd.random_normal(loc=0, scale=weight_scale, shape=(64, 2), ctx=ctx )    

    b1 = nd.random_normal(shape=int(16*nf), scale=weight_scale, ctx=ctx)
    b2 = nd.random_normal(shape=int(32*nf), scale=weight_scale, ctx=ctx)
    b3 = nd.random_normal(shape=int(64*nf), scale=weight_scale, ctx=ctx)
    b4 = nd.random_normal(shape=64, scale=weight_scale, ctx=ctx)    
    b5 = nd.random_normal(shape=64, scale=weight_scale, ctx=ctx)    
    b6 = nd.random_normal(shape=2, scale=weight_scale, ctx=ctx)        

    params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6]

    vs = []
    sqrs = []    
    
    # And assign space for gradients
    for param in params:
        param.attach_grad()
        vs.append(param.zeros_like())
        sqrs.append(param.zeros_like())        
    return params, vs, sqrs



if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)