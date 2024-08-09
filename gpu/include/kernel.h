#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
extern const int BLOCK_SIZE;
__global__ void initialize_gradient_(float *grad_bias, float *grad_weights, int M);

__global__ void find_delta_and_transpose(float *a, float *b, float *output, int N, int M);

__global__ void dev_vec_matmul(const float *dev_a, const float *dev_b, float *dev_output, int N, int M);

__global__ void gemm(const float *a, const float *b, const float *c, float *output, int N, int M, int K);

__global__ void matmul(const float *a, const float *b, float *output, int N, int M, int K);

__global__ void update_bias(float *dev_bias, float *layer_delta, int N, int M, float learning_rate);

__global__ void update_weights(float *dev_weights, float *layer_output, float *layer_delta, int N, int M, int K, float learning_rate);

__global__ void rmsprop_update_bias_(float *dev_bias, float *layer_delta, float *gradient_sum, int N, int M, float learning_rate, float beta); 

__global__ void rmsprop_update_weights_(float *dev_weights, float *layer_output, float *layer_delta, float *gradient_sum, int N, int M, int K, float learning_rate, float beta);

#endif
