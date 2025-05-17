#!/bin/bash
#SBATCH --job-name=compiling
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100-20:1
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out



# Load the correct CUDA version
module load cuda/11.7
module load cmake/3.24.2  


export CUDA_HOME=/public/lmod/cuda/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDNN_INCLUDE_DIR=/public/lmod/libs/cudnn-8.9.0/include
export CUDNN_LIBRARY=/public/lmod/libs/cudnn-8.9.0/lib/libcudnn.so
export CUDNN_LIBRARY_PATH=/public/lmod/libs/cudnn-8.9.0/lib
export LD_LIBRARY_PATH=$CUDNN_LIBRARY_PATH:$LD_LIBRARY_PATH

export CC=/opt/ohpc/pub/compiler/gcc/12.2.0/bin/gcc
export CXX=/opt/ohpc/pub/compiler/gcc/12.2.0/bin/g++
export CUDACXX=$CUDA_HOME/bin/nvcc

rm -rf build CMakeCache.txt

cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" \
  -DCUDNN_INCLUDE_DIR=/public/lmod/libs/cudnn-8.9.0/include \
  -DCUDNN_LIBRARY=/public/lmod/libs/cudnn-8.9.0/lib/libcudnn.so \
  -Dcudnn_INCLUDE_DIR=/public/lmod/libs/cudnn-8.9.0/include \
  -Dcudnn_LIBRARY=/public/lmod/libs/cudnn-8.9.0/lib/libcudnn.so \
  -Dcaffe2_USE_CUDNN=ON \
  -DCMAKE_PREFIX_PATH="/public/lmod/libs/cudnn-8.9.0/lib" \
  .

cmake --build build --target main --config RelWithDebInfo -j
