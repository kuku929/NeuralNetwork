#include "debug.h"
#include "kernel.h"
#include "layer.h"
#include "optimizer.h"
#include <memory>
#define BLOCK_SIZE 8
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

using namespace nnet;
RMSProp::RMSProp(float learning_rate, float beta) : Optimizer(learning_rate), beta(beta)
{
}

void RMSProp::initialize(const Shape &shape)
{
    // allocating device memory for gradient sum
    // will this be zero by default or should i launch a kernel to make them zero?
    dev_grad_weights_ = shared_dev_vector(shape.second, shape.first);
    dev_grad_bias_ = shared_dev_vector(shape.second, 1);

    // NOTE: this is not good! the input layer can be huge,
    // need to account for that
    dim3 dim_block(shape.first, 1);
    dim3 dim_grid(shape.second, 1);

    initialize_gradient_<<<dim_grid, dim_block>>>(dev_grad_bias_->data(), dev_grad_weights_->data(),
                                                  shape.first);
    cudaDeviceSynchronize();
}

std::shared_ptr<Optimizer> RMSProp::clone()
{
    auto optim = std::make_shared<RMSProp>(*this);
    return optim;
}

void RMSProp::update_bias(Layer *layer, const dev_vector<float> &layer_delta,
                          const size_t no_of_samples)
{

    // __START_TIMER__
    rmsprop_update_bias_<<<1, layer->nrows>>>(layer->dev_bias_->data(), layer_delta.data(),
                                              dev_grad_bias_->data(), no_of_samples, layer->nrows,
                                              learning_rate, beta);
    cudaDeviceSynchronize();
    // __END_TIMER__

    // auto result = cudaMemcpy(this->data(), dev_weights_->data(),
    // sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost); if(result != cudaSuccess){ throw
    // std::runtime_error("failed to copy to host!");
    //}
    // result = cudaMemcpy(this->bias(), dev_bias_->data(), sizeof(float)*dev_bias_->size(),
    // cudaMemcpyDeviceToHost); if(result != cudaSuccess){ throw std::runtime_error("failed to copy
    // to host!");
    //}

    return;
}

void RMSProp::update_weights(Layer *layer, const dev_vector<float> &layer_delta,
                             std::shared_ptr<dev_vector<float>> layer_output,
                             const size_t no_of_samples)
{

    // __START_TIMER__
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((layer->nrows + dim_block.x - 1) / dim_block.x,
                  (layer->ncols + dim_block.y - 1) / dim_block.y);
    // test_kernel<<<1,1>>>(dev_grad_weights_->data());
    // std::cout << "testing layer: " << this->layer->nrows <<  ' ' << this->layer->ncols << '\n';
    rmsprop_update_weights_<<<dim_grid, dim_block>>>(
        layer->dev_weights_->data(), layer_output->data(), layer_delta.data(),
        dev_grad_weights_->data(), layer->ncols, layer->nrows, no_of_samples, learning_rate, beta);
    cudaDeviceSynchronize();
    // __END_TIMER__

    // do i need to do this?
    //  auto result = cudaMemcpy(this->data(), dev_weights_->data(),
    //  sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost); if(result != cudaSuccess){
    //  	throw std::runtime_error("failed to copy to host!");
    //  }
    //  result = cudaMemcpy(this->bias(), dev_bias_->data(), sizeof(float)*dev_bias_->size(),
    //  cudaMemcpyDeviceToHost); if(result != cudaSuccess){ 	throw std::runtime_error("failed to
    //  copy to host!");
    //  }

    return;
}
