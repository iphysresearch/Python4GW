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

mx.random.seed(1)
random.seed(1)

def test_ctx():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    print('CPU or GPU? : ', ctx)
    
def lr_decay(lr, epoch, lr_decay):
    # 学习率自我衰减。
    if epoch > 2:
#         lr *= 0.1
        lr /= (1+lr_decay * epoch)
    return lr


# Mini-batch stochastic gradient descent.
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
    return params
        

# Adam.
def adam(params, vs, sqrs, lr, batch_size, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8
#     print(params)
    for param, v, sqr in zip(params, vs, sqrs):
#         print(param)
        g = params[param].grad / batch_size

        v[:] = beta1 * v + (1. - beta1) * g
        sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

        v_bias_corr = v / (1. - beta1 ** t)
        sqr_bias_corr = sqr / (1. - beta2 ** t)

        div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
        params[param][:] = params[param] - div
        
    return params, vs, sqrs


if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)