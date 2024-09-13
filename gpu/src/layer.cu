/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : definition of the Layer class
 */

#include "debug.h"
#include "kernel.h"
#include "layer.h"
#include "optimizer.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
using namespace nnet;
int SWITCH_OPTIMIZER = 0;
extern const int BLOCK_SIZE;

#if defined(DEBUG)
#define shared_dev_vector(a, b) shared_dev_vector(a, b, __LINE__, __FILE_NAME__)
#define set(a, b) set(a, b, __LINE__, __FILE_NAME__)
#endif

#define __START_TIMER__                                                                            \
    float gpu_elapsed_time_ms;                                                                     \
    cudaEvent_t start, stop;                                                                       \
    cudaEventCreate(&start);                                                                       \
    cudaEventCreate(&stop);                                                                        \
    cudaEventRecord(start, 0);

#define __END_TIMER__                                                                              \
    cudaEventRecord(stop, 0);                                                                      \
    cudaEventSynchronize(stop);                                                                    \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);                                       \
    std::cout << "gpu time : " << gpu_elapsed_time_ms << "ms\n";

Layer::Layer(int N, int M, Optimizer &optimizer) : basic_matrix(N, M), BaseLayer(M, N)
{
    bias_.resize(N);
    float range = 2.0f;
    // initializing random values
    for (size_t i = 0; i < dim.second; ++i)
    {
        this->get_bias(i) = 0.0f; // bias for the next layer
        for (size_t j = 0; j < dim.first; ++j)
        {
            float normalized_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) -
                                     0.5f; // between (-0.5,0.5)
            this->get_weights(i, j) = normalized_value * range;
        }
    }

    dev_weights_ = shared_dev_vector(dim.second, dim.first);
    dev_weights_->set(this->data(), dim.second * dim.first);
    cudaDeviceSynchronize();

    dev_bias_ = shared_dev_vector(dim.second, 1);
    dev_bias_->set(this->bias(), dim.second);
    // initializing optimizer
    m_optimizer = optimizer.clone();
    m_optimizer->initialize(dim);
}

std::shared_ptr<dev_vector<float>> Layer::forward_pass(
    const std::shared_ptr<dev_vector<float>> input, const size_t no_of_samples)
{
    /*
     * @brief forward pass of the input through the layer given,
     * calls gemm kernel to do parallel processing
     * using cache-tiled multiplication
     */

    // __START_TIMER__
    // copying data to gpu
    // dev_vector<float> weights(this);
    // dev_vector<float> bias(dim.second);
    // bias.set(this->bias(), dim.second);

    layer_input = input;
    layer_output = shared_dev_vector(dim.second, no_of_samples);

    // launching kernel
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((no_of_samples + dim_block.x - 1) / dim_block.x,
                  (dim.second + dim_block.y - 1) / dim_block.y);
    gemm<<<dim_grid, dim_block>>>(dev_weights_->data(), input->data(), dev_bias_->data(),
                                  layer_output->data(), dim.second, no_of_samples, dim.first);

    // __END_TIMER__

    return layer_output;
}

std::shared_ptr<dev_vector<float>> Layer::back_pass(const std::shared_ptr<dev_vector<float>> input,
                                                    const size_t no_of_samples)
{
    /*
     * @brief :
     * backward pass of the input through the layer given, stores output in
     * provided memory. calls matmul kernel to do parallel processing
     * using cache-tiled multiplication
     * once pass is done, updates the layer's weights and bias
     */
    auto back_output = shared_dev_vector(get_shape().first, no_of_samples);
    // launching kernel
    const size_t input_cols = dim.second;
    const size_t input_rows = no_of_samples;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((dim.first + dim_block.x - 1) / dim_block.x,
                  (input_rows + dim_block.y - 1) / dim_block.y);
    matmul<<<dim_grid, dim_block>>>(input->data(), dev_weights_->data(), back_output->data(),
                                    input_rows, dim.first, input_cols);

    cudaDeviceSynchronize();
    update(input, no_of_samples);
    return back_output;
}

void Layer::update(const std::shared_ptr<dev_vector<float>> &layer_delta,
                   const size_t no_of_samples)
{
    /*
     * @brief :
     * uses update_bias() and update_weights if SWITCH_OPTIMIZER set,
     * else uses the provided optimizer. Useful if you
     * want to switch optimizers while training
     */
    //  __START_TIMER__
    float learning_rate = 0.09f;
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((dim.second + dim_block.x - 1) / dim_block.x,
                  (dim.first + dim_block.y - 1) / dim_block.y);

    if (SWITCH_OPTIMIZER)
    {
        update_weights<<<dim_grid, dim_block>>>(dev_weights_->data(), layer_input->data(),
                                                layer_delta->data(), dim.first, dim.second,
                                                no_of_samples, learning_rate);
        update_bias<<<1, dim.second>>>(dev_bias_->data(), layer_delta->data(), no_of_samples,
                                       dim.second, learning_rate);
        cudaDeviceSynchronize();
    }
    else
    {
        m_optimizer->update_weights(this, *layer_delta, layer_input, no_of_samples);
        m_optimizer->update_bias(this, *layer_delta, no_of_samples);
    }

#if defined(DEBUG)
    copy_to_host();
#endif
    // __END_TIMER__
    return;
}

void Layer::copy_to_host()
{
    auto result = cudaMemcpy(this->data(), this->dev_weights_->data(),
                             sizeof(float) * dev_weights_->size(), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host!");
    }
    result = cudaMemcpy(this->bias(), dev_bias_->data(), sizeof(float) * dev_bias_->size(),
                        cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host!");
    }
}
