#include "dev_vector.h"
#include "layer.h"
#include "kernel.h"
#include <vector>
#include <algorithm>
#include <array>
#include <memory>
#include <iostream>
#include <cstdio>
using namespace nnet;
extern const int BLOCK_SIZE;
#define DEBUG

// #define __START_TIMER__  float gpu_elapsed_time_ms; \
// 	cudaEvent_t start, stop; \
// 	cudaEventCreate(&start); \
// 	cudaEventCreate(&stop); \
// 	cudaEventRecord(start, 0); \

// #define __END_TIMER__ cudaEventRecord(stop, 0); \
// 	cudaEventSynchronize(stop); \
// 	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop); \
// 	std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n"; \


Layer::Layer(int N, int M): 
	basic_matrix(N,M),
	dim(M, N)
{
	bias_.resize(N);	
	float range = 2.0f;
	//initializing random values 
	//todo : make initialization gpu
	for(size_t i=0;i<dim.second;++i){
		this->get_bias(i) = 0.0f; //bias for the next layer
		for(size_t j=0;j<dim.first;++j){
			float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
			this->get_weights(i, j) = normalized_value*range; 
		}
	}

	dev_weights_ = std::make_shared<dev_vector<float>>(dim.second*dim.first);
	dev_weights_->set(this->data(), dim.second*dim.first);
	cudaDeviceSynchronize();

	dev_bias_ = std::make_shared<dev_vector<float>>(dim.second);
	dev_bias_->set(this->bias(), dim.second);


}

void Layer::forward_pass(const dev_vector<float> &input, dev_vector<float> &output, const size_t no_of_samples){
/*
 * @argument : Layer object
 *		input to layer
 *		dev_vector to store the output( memory should be allocated )
 *		no of samples to be processed
 * @brief : 
 * forward pass of the input through the layer given, stores output in provided memory.
 * calls gemm_function kernel to do parallel processing
 * using cache-tiled multiplication
 */

	// __START_TIMER__
	//copying data to gpu 
	//dev_vector<float> weights(this);
	//dev_vector<float> bias(dim.second);
	//bias.set(this->bias(), dim.second);
	

	//launching kernel
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((no_of_samples + dim_block.x - 1)/dim_block.x, (dim.second + dim_block.y - 1)/dim_block.y);
	gemm<<<dim_grid , dim_block>>>(dev_weights_->data(), input.data(), dev_bias_->data(), output.data(), dim.second, no_of_samples, dim.first);

	// //debug
	// std::cout << "forward_pass weights:\n";
	// basic_matrix<float> h_output(dim.second, dim.first);
	// cudaMemcpy(h_output.data(), dev_weights_->data(), sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost);
	// h_output.show();
	// std::cout << "forward_pass weights end--\n";
	// cudaDeviceSynchronize();

	// __END_TIMER__
	return;

}

void Layer::back_pass(const dev_vector<float> &input, dev_vector<float> &output, const size_t no_of_samples)
{
/*
 * @argument : Layer object
 *		input to layer of dim(no_of_samples, layer.dim.first)
 *		dev_vector to store the output( memory should be allocated )
 * 		layer output of the back-nodes
 *		no of samples to be processed
 * @brief : 
 * backward pass of the input through the layer given, stores output in provided memory.
 * calls matmul_funcmul kernel to do parallel processing
 * using cache-tiled multiplication
 * once pass is done, updates the layer's weights and bias using layer_output and input
 * by calling update_layer()
 */
	//copying data to gpu 
	//dev_vector<float> weights(this);
	//dev_vector<float> bias(dim.second);
	//bias.set(this->bias(), dim.second);
	
	//launching kernel
	const size_t input_cols = dim.second;
	const size_t input_rows = no_of_samples;
	
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((dim.first + dim_block.x - 1)/dim_block.x, (input_rows + dim_block.y - 1)/dim_block.y);
	matmul<<<dim_grid, dim_block>>>(input.data(), dev_weights_->data(), output.data(), input_rows, dim.first, input_cols);	
	cudaDeviceSynchronize();
	//note : do not forget that the output matrix is transposed (no_of_samples, cols)
	//todo : maybe layer_output should be stored in the layer itself?
	return;
}


void Layer::update(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples){
/* @arguments : Layer object
 * 		weights in device memory
 * 		bias in device memory
 *		deltas for that layer
 * 		output of the previous layer during forward pass
 * 		no of samples being trained concurrently
 *
 * @brief : 
 * calls two kernels, update_weights() and update_bias() to update the layer
 * cache-tiled multiplication used to update the weights
 * after updating, the new weights and bias are copied to host
 */
	//  __START_TIMER__
	float learning_rate = 0.09f;
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((dim.second + dim_block.x - 1)/dim_block.x, (dim.first + dim_block.y - 1)/dim_block.y);
	update_weights<<<dim_grid, dim_block>>>(dev_weights_->data(), layer_output->data(), layer_delta.data(), dim.first, dim.second, no_of_samples, learning_rate); 
	update_bias<<<1, dim.second>>>(dev_bias_->data(), layer_delta.data(), no_of_samples, dim.second, learning_rate); 
	cudaDeviceSynchronize();

	#if defined(DEBUG)  
	copy_to_host();
	#endif
	// __END_TIMER__
	return;
}

void Layer::copy_to_host()
{
	// do i need to do this?
	auto result = cudaMemcpy(this->data(), this->dev_weights_->data(), sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost);
	if(result != cudaSuccess){
		throw std::runtime_error("failed to copy to host!");
	}
	result = cudaMemcpy(this->bias(), dev_bias_->data(), sizeof(float)*dev_bias_->size(), cudaMemcpyDeviceToHost);
	if(result != cudaSuccess){
		throw std::runtime_error("failed to copy to host!");
	}
}
