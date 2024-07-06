#include "dev_vector.h"
#include "layer.h"
#include "kernel.h"
#include <vector>
#include <algorithm>
#include <array>
#include <memory>
#include <iostream>
#include <cstdio>
using namespace layer;
extern const int BLOCK_SIZE;

Layer::Layer(int N, int M): basic_matrix(N,M){
	bias_.resize(N);	

	float range = 2.0f;
	//initializing random values 
	//todo : make initialization gpu
	for(size_t i=0;i<nrows;++i){
		this->get_bias(i) = 0.0f; //bias for the next layer
		for(size_t j=0;j<ncols;++j){
			float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
			this->get_weights(i, j) = normalized_value*range; 
		}
	}


	dev_weights_ = std::make_shared<dev_vector<float>>(nrows*ncols);
	dev_weights_->set(this->data(), nrows*ncols);
	cudaDeviceSynchronize();

	dev_bias_ = std::make_shared<dev_vector<float>>(nrows);
	dev_bias_->set(this->bias(), nrows);


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

	//float gpu_elapsed_time_ms;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	//copying data to gpu 
	//dev_vector<float> weights(this);
	//dev_vector<float> bias(nrows);
	//bias.set(this->bias(), nrows);
	

	//launching kernel
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((no_of_samples + dim_block.x - 1)/dim_block.x, (nrows + dim_block.y - 1)/dim_block.y);
	gemm<<<dim_grid , dim_block>>>(dev_weights_->data(), input.data(), dev_bias_->data(), output.data(), nrows, no_of_samples, ncols);

	cudaDeviceSynchronize();

	////timing 
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";

	return;

}

void Layer::back_pass(const dev_vector<float> &input, dev_vector<float> &output, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples){
/*
 * @argument : Layer object
 *		input to layer of dim(no_of_samples, layer.ncols)
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
	//float gpu_elapsed_time_ms;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	//copying data to gpu 
	//dev_vector<float> weights(this);
	//dev_vector<float> bias(nrows);
	//bias.set(this->bias(), nrows);
	
	//launching kernel
	const size_t input_cols = nrows;
	const size_t input_rows = no_of_samples;
	
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((ncols + dim_block.x - 1)/dim_block.x, (input_rows + dim_block.y - 1)/dim_block.y);
	//todo: implement backwards activation
	matmul<<<dim_grid , dim_block>>>(input.data(), dev_weights_->data(), output.data(), input_rows, ncols, input_cols);	
	cudaDeviceSynchronize();
	//note : forget that the output matrix is transposed (no_of_samples, cols)

	////timing 
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";

	//todo : maybe layer_output should be stored in the layer itself?
	update(input, layer_output, no_of_samples);
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
	//float gpu_elapsed_time_ms;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	float learning_rate = 0.09f;
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((nrows + dim_block.x - 1)/dim_block.x, (ncols + dim_block.y - 1)/dim_block.y);
	update_weights<<<dim_grid, dim_block>>>(dev_weights_->data(), layer_output->data(), layer_delta.data(), ncols, nrows, no_of_samples, learning_rate); 
	update_bias<<<1, nrows>>>(dev_bias_->data(), layer_delta.data(), no_of_samples, nrows, learning_rate); 
	cudaDeviceSynchronize();

	////do i need to do this?
	//auto result = cudaMemcpy(this->data(), dev_weights_.data(), sizeof(float)*dev_weights_.size(), cudaMemcpyDeviceToHost);
	//if(result != cudaSuccess){
		//throw std::runtime_error("failed to copy to host!");
	//}

	//result = cudaMemcpy(this->bias(), dev_bias_.data(), sizeof(float)*dev_bias_.size(), cudaMemcpyDeviceToHost);
	//if(result != cudaSuccess){
		//throw std::runtime_error("failed to copy to host!");
	//}

	////timing 
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";

	return;
}
