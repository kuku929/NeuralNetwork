#include "optimizer.h"
#include "layer.h"
#include "kernel.h"
#define BLOCK_SIZE 8
#define __START_TIMER__  float gpu_elapsed_time_ms; \
	cudaEvent_t start, stop; \
	cudaEventCreate(&start); \
	cudaEventCreate(&stop); \
	cudaEventRecord(start, 0); \

#define __END_TIMER__ cudaEventRecord(stop, 0); \
	cudaEventSynchronize(stop); \
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop); \
	std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n"; \

using namespace optimizer;
RMSProp::RMSProp(std::shared_ptr<layer::Layer> layer, float learning_rate, float beta): Optimizer(layer, learning_rate), beta(beta){ 
	//allocating device memory for gradient sum
	//will this be zero by default or should i launch a kernel to make them zero?
	dev_grad_weights_ = std::make_shared<dev_vector<float>>(layer->nrows*layer->ncols);
	dev_grad_bias_ = std::make_shared<dev_vector<float>>(layer->nrows);

	dim3 dim_block(layer->ncols, 1);
	dim3 dim_grid(layer->nrows, 1);


	// std::cout << "testing layer 1 : " << this->layer->nrows <<  ' ' << this->layer->ncols << '\n';
	initialize_gradient_<<<dim_grid, dim_block>>>(dev_grad_bias_->data(), dev_grad_weights_->data(), layer->ncols);
}

void RMSProp::update_bias(const dev_vector<float> &layer_delta, const size_t no_of_samples){

	// __START_TIMER__
	rmsprop_update_bias_<<<1, layer->nrows>>>(layer->dev_bias_->data(), layer_delta.data(), dev_grad_bias_->data(), no_of_samples, layer->nrows, learning_rate, beta); 
	cudaDeviceSynchronize();
	// __END_TIMER__

	//do i need to do this?
	// auto result = cudaMemcpy(this->data(), dev_weights_->data(), sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost);
	// if(result != cudaSuccess){
	// 	throw std::runtime_error("failed to copy to host!");
	// }
	// result = cudaMemcpy(this->bias(), dev_bias_->data(), sizeof(float)*dev_bias_->size(), cudaMemcpyDeviceToHost);
	// if(result != cudaSuccess){
	// 	throw std::runtime_error("failed to copy to host!");
	// }


	return;
}

__global__ void test_kernel(float * t)
{
	printf("testing gradient : %f\n", t[0]);
}

void RMSProp::update_weights(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples){

	// __START_TIMER__
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((layer->nrows + dim_block.x - 1)/dim_block.x, (layer->ncols + dim_block.y - 1)/dim_block.y);
	// test_kernel<<<1,1>>>(dev_grad_weights_->data());
	// std::cout << "testing layer: " << this->layer->nrows <<  ' ' << this->layer->ncols << '\n';
	rmsprop_update_weights_<<<dim_grid, dim_block>>>(layer->dev_weights_->data(), layer_output->data(), layer_delta.data(), dev_grad_weights_->data(), layer->ncols, layer->nrows, no_of_samples, learning_rate, beta); 
	cudaDeviceSynchronize();
	// __END_TIMER__

	//do i need to do this?
	// auto result = cudaMemcpy(this->data(), dev_weights_->data(), sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost);
	// if(result != cudaSuccess){
	// 	throw std::runtime_error("failed to copy to host!");
	// }
	// result = cudaMemcpy(this->bias(), dev_bias_->data(), sizeof(float)*dev_bias_->size(), cudaMemcpyDeviceToHost);
	// if(result != cudaSuccess){
	// 	throw std::runtime_error("failed to copy to host!");
	// }


	return;
}