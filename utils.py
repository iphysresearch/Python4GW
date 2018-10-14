#! usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function

# importing MxNet >= 1.0
import mxnet as mx
import mxnet.ndarray as nd

# Data manipulation
# import numpy as np
# import pandas as pd

import os, sys
try: import seaborn as sns
except:  os.system('pip install seaborn') ; os.system('pip install --upgrade pip') ;import seaborn as sns

def check_ctx():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except Exception as e:
        ctx = mx.cpu()
#         print(e)
    return ctx


def check_dict_dim(d):
    """
    检查字典中每一个键里值的个数是否都是相同的维度。
    """
    assert len([value for index, values in enumerate(d.values()) \
                if index == 0 \
                for index, value in enumerate(d.values()) \
                if len(values) == len(value) ])  == len(d.keys()), "1123"


def get_variable_name(variable, loc):
    """
    获得变量的str名称
    REF: https://blog.csdn.net/Yeoman92/article/details/75076166
    """
    # loc = locals()
    # print(loc)
    for key in loc:
        if loc[key] == variable:
            return key

import inspect
def retrieve_name(var):
    '''
    utils:
    get back the name of variables
    '''
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    try:
        output = [var_name for var_name, var_val in callers_local_vars if var_val is var]
        assert len(output) == 1
    except:
        print('Found same value in:', output)
    finally:
        return output[0]

def mkdir_checkdir(path = "/output"):

    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('MKDIR: ' + path + ' successful!')
    else:
        print(path + " have existed!")
        
