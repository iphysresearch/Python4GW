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


# Non-linear function.
def relu(X):
    return nd.maximum(X,nd.zeros_like(X))


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




def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃。
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape, ctx=ctx) < keep_prob
    return mask * X / keep_prob



if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)