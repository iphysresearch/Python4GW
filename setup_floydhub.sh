#!/bin/bash
# The script that help make my code ready
# for fun in Floydhub.com

# update pip
pip install --upgrade pip

# install Mxnet for GPU
pip install -U --pre mxnet-cu91
sudo ldconfig /usr/local/cuda-9.1/lib64

# install line-profiler
pip install line-profiler



