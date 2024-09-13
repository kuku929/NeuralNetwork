/*
 *@Author: Krutarth Patel                                           
 *@Date: 13th september 2024
 *@Description : definition of the ActivationLayer class
 */

#include "activation.h"
#include "dev_vector.h"
#include <memory>

using namespace nnet;

template <typename f_func_ptr>
__global__ void f_activate(const float *input, float *output, int N, int M,
                           f_func_ptr activ_func)
{
    /*
     * @brief : templated forward activation function kernel for forward pass
     * @note :
     * input is NxM
     * M is expected to be no of samples
     * each sample will be evaluated by a block
     * thus, threadId corresponds to row
     */
    int row_in_matrix = threadIdx.x; // row corresponds to node
    int col_in_matrix = blockIdx.x;  // col corresponds to sample
    int index = row_in_matrix * M + col_in_matrix;
    if (row_in_matrix < N && col_in_matrix < M)
    {
        output[index] = activ_func(input[index]);
    }
}

template <typename b_func_ptr>
__global__ void b_activate(const float *input, float *layer_output,
                           float *output, int N, int M, b_func_ptr activ_func)
{
    /*
     * @brief : templated backward activation function kernel for back pass
     * @note :
     * each sample will be evaluated by a block
     * thus, threadId corresponds to column
     * input is NxM
     * layer output is MxN
     */
    int row_in_matrix = blockIdx.x;  // row should correspond to the sample
    int col_in_matrix = threadIdx.x; // col should correspond to the node
    int index = row_in_matrix * M + col_in_matrix;
    int t_index = col_in_matrix * N + row_in_matrix;
    if (row_in_matrix < N && col_in_matrix < M)
    {
        output[index] = activ_func(input[index], layer_output[t_index]);

    }

}

dev_ptr Linear::forward_pass(const dev_ptr input, size_t no_of_samples)
{
    layer_output = input;
    layer_input = input;
    return input;
}

dev_ptr Linear::back_pass(const dev_ptr input, const size_t no_of_samples)
{
    /*
     * @note :
     * input is (no of samples, size)
     * layer_output is (size, no of samples)
     * output is (no of samples, size)
     */
    return input;
}

dev_ptr Sigmoid::forward_pass(const dev_ptr input, size_t no_of_samples)
{
    /*
     * @note :
     * input is (size, no of samples)
     */
    layer_input = input;
    layer_output =
        std::make_shared<dev_vector<float>>(dim.second * no_of_samples);
    dim3 dim_block(dim.second, 1);
    dim3 dim_grid(no_of_samples, 1);
    f_activate<<<dim_grid, dim_block>>>(input->data(), layer_output->data(),
                                        dim.second, no_of_samples, f_sigmoid_);
    cudaDeviceSynchronize();
    return layer_output;
}

dev_ptr Sigmoid::back_pass(const dev_ptr input, const size_t no_of_samples)
{
    /*
     * @note :
     * input is (no of samples, size)
     * layer_output is (size, no of samples)
     * output is (no of samples, size)
     */
    dim3 dim_block(dim.second, 1);
    dim3 dim_grid(no_of_samples, 1);
    b_activate<<<dim_grid, dim_block>>>(input->begin(), layer_output->data(),
                                        layer_input->data(), no_of_samples,
                                        dim.second, b_sigmoid_);
    cudaDeviceSynchronize();
    return layer_input;
}

dev_ptr ReLU::forward_pass(const dev_ptr input, size_t no_of_samples)
{
    /*
     * @note :
     * input is (size, no of samples)
     */
    layer_input = input;
    layer_output =
        std::make_shared<dev_vector<float>>(dim.second * no_of_samples);
    dim3 dim_block(dim.second, 1);
    dim3 dim_grid(no_of_samples, 1);
    f_activate<<<dim_grid, dim_block>>>(input->data(), layer_output->data(),
                                        dim.second, no_of_samples, f_relu_);
    cudaDeviceSynchronize();
    return layer_output;
}

dev_ptr ReLU::back_pass(const dev_ptr input, const size_t no_of_samples)
{
    /*
     * @note :
     * input is (no of samples, size)
     * layer_output is (size, no of samples)
     * output is (no of samples, size)
     */
    dim3 dim_block(dim.second, 1);   // a block will evaluate one sample
    dim3 dim_grid(no_of_samples, 1); // one block per sample
    b_activate<<<dim_grid, dim_block>>>(input->begin(), layer_output->data(),
                                        layer_input->data(), no_of_samples,
                                        dim.second, b_relu_);
    cudaDeviceSynchronize();
    return layer_input;
}
