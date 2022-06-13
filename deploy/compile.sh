#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "/workspace/cadnn/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/c++/lib"
  exit 1
fi

# 2. compile
mkdir -p build
cd build

DEMO_NAME=caddn_main

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON
WITH_ONNXRUNTIME=ON

LIB_DIR=/workspace/cadnn/paddle_inference

CUDNN_LIB=/usr/local/x86_64-pc-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-6.0.1.8

# CUSTOM_OPERATOR_FILES="custom_relu_op.cc;custom_relu_op.cu"
CUSTOM_OPERATOR_FILES="grid_sample_3d.cc;grid_sample_3d.cu"


cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES} \
  -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME}

make -j
