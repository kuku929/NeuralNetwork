#pragma once
#include <stdio.h>
extern const int BLOCK_SIZE;
__global__ void find_delta_and_transpose(float *a, float *b, float *output, int N, int M);

__global__ void dev_vec_matmul(const float *dev_a, const float *dev_b, float *dev_output, int N, int M);

__global__ void gemm(const float *a, const float *b, const float *c, float *output, int N, int M, int K);

__global__ void matmul(const float *a, const float *b, float *output, int N, int M, int K);

__global__ void update_bias(float *dev_bias, float *layer_delta, int N, int M, float learning_rate); 

__global__ void update_weights(float *dev_weights, float *layer_output, float *layer_delta, int N, int M, int K, float learning_rate);
