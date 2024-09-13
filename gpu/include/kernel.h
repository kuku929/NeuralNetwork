/*
 *@Author: Krutarth Patel                                           
 *@Date: 13th september 2024
 *@Description : declaration of all kernels
 */

#ifndef KERNEL_H
#define KERNEL_H

extern const int BLOCK_SIZE;
__global__ void initialize_gradient_(float *grad_bias, float *grad_weights, int M);
/*
 * @brief set rmsprop gradients to zero
 * @params bias, weights, if the layer size is (first, second)
 * 			M = first, here is an example:
 * #\
 *   \ 
 * #--# 	size = (3,1)
 *   / 		M = first = 3
 * #/
 *
 */

__global__ void dev_vec_matmul(const float *dev_a, const float *dev_b, float *dev_output, int N, int M);
/*
 * @brief multiply a vector with a matrix 
 * @params dev_a is a matrix, dev_b is a vector
 * 			dev_a has the dimensions (row, col) = (N, M)
 *
 */

__global__ void gemm(const float *a, const float *b, const float *c, float *output, int N, int M, int K);
/*
 * @brief performs general matrix multiplication, i.e. A*B + C = output
 * @params a is a matrix, b is another matrix, c is a vector 
 * 			a has the dimensions (row, col) = (N, K)
 * 			b has the dimensions (row, col) = (K, M)
 * 			c has the dimensions (row, col) = (N, 1)
 * 			output has the dimensions (row, col) = (N, M)
 *
 */

__global__ void matmul(const float *a, const float *b, float *output, int N, int M, int K);
/*
 * @brief performs matrix multiplication
 * @params a is a matrix, b is another matrix
 * 			a has the dimensions (row, col) = (N, K)
 * 			b has the dimensions (row, col) = (K, M)
 * 			output has the dimensions (row, col) = (N, M)
 *
 */

__global__ void update_bias(float *dev_bias, float *layer_delta, int N, int M, float learning_rate);
/*
 * @brief updates biases by backpropagation
 * @params dev_bias: bias, layer_delta: propagated delta
 * 			dev_bias has the dimensions (row, col) = (M, 1)
 * 			layer_delta has the dimensions (row, col) = (N, M)
 *
 */

__global__ void update_weights(float *dev_weights, float *layer_output, float *layer_delta, int N, int M, int K, float learning_rate);
/*
 * @brief updates weights by backpropagation
 * @params dev_weights: weights, layer_outputs: output of previos layer during forward_pass, layer_delta: propagated delta
 * 			dev_weights has the dimensions (row, col) = (N, M)
 * 			layer_output has the dimensions (row, col) = (N, K)
 * 			layer_delta has the dimensions (row, col) = (K, M)
 *
 */

__global__ void rmsprop_update_bias_(float *dev_bias, float *layer_delta, float *gradient_sum, int N, int M, float learning_rate, float beta); 
/*
 * @brief updates biases by backpropagation
 * @params dev_bias: bias, layer_delta: propagated delta, gradient_sum: bias gradients
 * 			dev_bias has the dimensions (row, col) = (M, 1)
 * 			gradient_sum has the dimensions (row, col) = (M, 1)
 * 			layer_delta has the dimensions (row, col) = (N, M)
 *
 */

__global__ void rmsprop_update_weights_(float *dev_weights, float *layer_output, float *layer_delta, float *gradient_sum, int N, int M, int K, float learning_rate, float beta);
/*
 * @brief updates weights by backpropagation
 * @params dev_weights: weights, layer_outputs: output of previos layer during forward_pass, layer_delta: propagated delta, gradient_sum: weight gradients
 * 			dev_weights has the dimensions (row, col) = (N, M)
 * 			gradient_sum has the dimensions (row, col) = (N, M)
 * 			layer_output has the dimensions (row, col) = (N, K)
 * 			layer_delta has the dimensions (row, col) = (K, M)
 *
 */
#endif
