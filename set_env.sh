#! /bin/bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/.conda/envs/zhong/lib:$LD_LIBRARY_PATH