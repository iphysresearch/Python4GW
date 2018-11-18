#!usr/bin/python
#coding=utf-8

import sys, os
sys.path.append(os.path.abspath(''))   # 把当前目录设为引用模块的地址之一

from data_utils import *
from data_noise import *

import mxnet as mx
import mxnet.ndarray as nd

ctx = check_ctx()

shape = (10000, 8192)

b = nd.array(pre_fir().reshape((-1,1)), ctx=ctx)
pp = pre_fftfilt(b, shape = shape, nfft=None)
print('Generating noise1...')
noise1 = GenNoise_matlab_nd(shape = shape, params = pp).asnumpy()
print('Generating noise2...')
noise2 = GenNoise_matlab_nd(shape = shape, params = pp).asnumpy()

# save
save_address1 = 'LigoNose9_9000_8192_3'
save_address2 = 'LigoNose9_9000_8192_4'
os.system('rm -rf ./*')

print('Saving noise1...')
np.save(save_address1, noise1)
print('Saving noise2...')
np.save(save_address2, noise2)

# floyd run --gpu --follow \
# -m "Gen_local_noises" \
# "bash setup_floydhub.sh && python gen_local_noises.py"