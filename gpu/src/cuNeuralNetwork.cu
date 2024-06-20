#include "Layer.h"
#include "cuNeuralNetwork.h"
#include "dev_vector.h"
#include "kernel.cuh"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath> //pow



void Network::add_layer(size_t next_layer_size, std::string next_activ_func){
 /*
  * @arguments : size of the next layer, activation function for the next layer
  * 
  * @brief : 
  * this function constructs the layer with a size of next_layer_size
  * and an ID number to activation function, it goes like this:
  * sigmoid, ReLU, Leaky, Linear
  *    0       1     2      3
  * if the max_layer_size is 0 while calling, it assumes added layer is
  * the first layer i.e. input layer.
  * 
  * after construction of Layer object, it is added to the vector layers
  *
  * max_layer_size is required for later use in code while allocating 
  * device memory.
  */
	//if layers have already been added
	//max layer size is zero when object initialized
	if(next_layer_size == 0){
		throw std::invalid_argument("layer cannot have 0 neurons");
	}

	if(max_layer_size){
		//getting the size of the last layer
		size_t last_layer_size;
		if(layers.size() == 0){
			last_layer_size = max_layer_size;
		}
		else
			last_layer_size = layers.back().nrows;

		int next_activ_func_id = std::distance(activation_functions_.begin(), std::find(activation_functions_.begin(), activation_functions_.end(), next_activ_func));
		if(layers.empty())
			//creating the first layer
			layers.push_back(Layer(next_layer_size, last_layer_size, next_activ_func_id));
		
		else
			//subsequent layers
			layers.push_back(Layer(next_layer_size, last_layer_size, next_activ_func_id, layers.back()));
		

		if(next_layer_size > max_layer_size)
			max_layer_size = next_layer_size;
		return;
	}
	
	if(next_activ_func != "Linear"){
		throw std::invalid_argument("first layer cannot have activation function");
	}
	//if no layers have been added yet, store the size of the layer
	max_layer_size = next_layer_size;
	return;
}


void pass(const Layer &layer, const dev_vector<float> &input, dev_vector<float> &output, const size_t no_of_samples){
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
	dev_vector<float> weights(layer);
	dev_vector<float> bias(layer.nrows);
	bias.set(layer.bias(), layer.nrows);
	
	//launching kernel
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((no_of_samples + dim_block.x - 1)/dim_block.x, (layer.nrows + dim_block.y - 1)/dim_block.y);
	gemm_function<<<dim_grid , dim_block>>>(weights.data(), input.data(), bias.data(), output.data(), layer.nrows, no_of_samples, layer.ncols, layer.activation_function);

	cudaDeviceSynchronize();

	////timing 
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";
	//return;
}

basic_matrix<float> Network::forward_pass(basic_matrix<float> &input){
/*
 * @arguments : input matrix with dimensions : (input_layer_size, no_of_samples)
 * 
 * @brief:
 * iterates through layers calling the pass() function on each one
 * the dev_input and dev_output are swapped as the input and output
 * to the layers to avoid copying. 
 * therefore, they are malloc'd with the maximum size at the beginning
 * Once the output is ready, it is copied to host
 */

	if(input.nrows != layers.front().ncols){
		throw std::invalid_argument("dimensions do not match!");
	}

	const size_t no_of_samples = input.ncols;

	//copying to device
	dev_vector<float> dev_input((max_layer_size)*no_of_samples);	
	dev_input.set(input.data(), input.size);
	dev_vector<float> dev_output((max_layer_size)*no_of_samples);	
	
	//forward pass
	int count = 0;
	for(const auto &layer : layers){
		if(count%2==0)
			pass(layer, dev_input, dev_output, no_of_samples);
		else
			pass(layer, dev_output, dev_input, no_of_samples);
		count++;
	}
	basic_matrix<float> output(layers.back().nrows, no_of_samples);

	//copying to host
	if(count%2 == 0){
		auto result = cudaMemcpy(output.data(), dev_input.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
		if(result != cudaSuccess){
			throw std::runtime_error("failed to copy to host!");
		}
	}
	else{
		auto result = cudaMemcpy(output.data(), dev_output.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
		if(result != cudaSuccess){
			throw std::runtime_error("failed to copy to host!");
		}
	}
	return output;

}

void update_layer(Layer &layer, dev_vector<float> &dev_weights, dev_vector<float> &dev_bias, const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples){
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
	dim3 dim_grid((layer.nrows + dim_block.x - 1)/dim_block.x, (layer.ncols + dim_block.y - 1)/dim_block.y);
	update_weights<<<dim_grid, dim_block>>>(dev_weights.data(), layer_output->data(), layer_delta.data(), layer.ncols, layer.nrows, no_of_samples, learning_rate); 
	update_bias<<<1, layer.nrows>>>(dev_bias.data(), layer_delta.data(), no_of_samples, layer.nrows, learning_rate); 
	cudaDeviceSynchronize();

	auto result = cudaMemcpy(layer.data(), dev_weights.data(), sizeof(float)*dev_weights.size(), cudaMemcpyDeviceToHost);
	if(result != cudaSuccess){
		throw std::runtime_error("failed to copy to host!");
	}

	result = cudaMemcpy(layer.bias(), dev_bias.data(), sizeof(float)*dev_bias.size(), cudaMemcpyDeviceToHost);
	if(result != cudaSuccess){
		throw std::runtime_error("failed to copy to host!");
	}

	////timing 
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";

	return;
}

void propagate(Layer &layer, const dev_vector<float> &input, dev_vector<float> &output, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples){
/*
 * @argument : Layer object
 *		input to layer
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
	dev_vector<float> weights(layer);
	dev_vector<float> bias(layer.nrows);
	bias.set(layer.bias(), layer.nrows);
	
	//launching kernel
	const size_t input_cols = layer.nrows;
	const size_t input_rows = no_of_samples;
	
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((layer.ncols + dim_block.x - 1)/dim_block.x, (input_rows + dim_block.y - 1)/dim_block.y);
	matmul_funcmul<<<dim_grid , dim_block>>>(input.data(), weights.data(), layer_output->data(), output.data(), input_rows, layer.ncols, input_cols, layer.back_activation_derivative);	
	cudaDeviceSynchronize();

	////timing 
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	//std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";

	update_layer(layer, weights, bias, input, layer_output, no_of_samples);
	return;
}

void Network::backward_pass(basic_matrix<float> &input, basic_matrix<float> &true_output){
/*
 * @arguments : input matrix with dimensions : (input_layer_size, no_of_samples)
 *		true output with dimensions : (output_layer_size, no_of_samples)
 * 
 * @brief:
 * finds the outputs of all the layers and stores them in layer_outputs
 * then, iterates backwards through layers calling the propagate() function on each one
 * the dev_input_delta and dev_output_delta are swapped as the input and output
 * to the layers to avoid copying. 
 * therefore, they are malloc'd with the maximum size at the beginning
 */
	/*
	 * the column in input will be for one training sample, we will do input.ncols number of samples together
	 */
	const size_t no_of_samples = input.ncols;
	//index i holds the layer outputs for layer i, each dev_vector will be a matrix with rows = layers.nrows, cols = no of samples 
	std::vector<std::shared_ptr<dev_vector<float>>> layer_outputs(layers.size()+1);

	//debug
	std::vector<basic_matrix<float>> h_layer_outputs(layers.size()+1);
	h_layer_outputs[0] = input;

	//adding the initial input
	layer_outputs[0] = std::make_shared<dev_vector<float>>(input);	
	//for the hidden layers
	for(int j=0; j < layers.size(); ++j){
		layer_outputs[j+1] = std::make_shared<dev_vector<float>>((layers[j].nrows)*no_of_samples);
		
		//debug
		h_layer_outputs[j+1] = basic_matrix<float>(layers[j].nrows, no_of_samples);

	}

	//forward pass
	for(int i=0; i < layers.size(); ++i){
		pass(layers[i], *layer_outputs[i], *layer_outputs[i+1], no_of_samples);
	}
	
	//debug
	for(int i=0; i < layer_outputs.size(); ++i){
		cudaMemcpy(h_layer_outputs[i].data(), layer_outputs[i]->data(), sizeof(float)*layer_outputs[i]->size(), cudaMemcpyDeviceToHost); 
	}
	std::cout << "LAYER OUTPUTS\n";
	for(auto const a : h_layer_outputs){
		a.show();
		std::cout << '\n';
	}

	//finding intial delta
	dev_vector<float> dev_true_output(true_output);

	//allocating maximum possible data to prevent needless copying
	dev_vector<float> dev_input_delta(no_of_samples*max_layer_size);
	dev_vector<float> dev_output_delta(no_of_samples*max_layer_size);


	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((true_output.ncols + dim_block.x - 1)/dim_block.x, (true_output.nrows + dim_block.y - 1)/dim_block.y);

	//finding the difference and transposing the resulting matrix
	//delta will have no_of_samples rows and layer size cols
	find_delta_and_transpose<<<dim_grid, dim_block>>>(dev_true_output.data(), (*layer_outputs.back()).data(), dev_input_delta.data(), true_output.nrows, true_output.ncols, layers.back().activation_derivative);


	//debug
	std::cout << "DELTA\n";
	std::vector<float> o(layers.back().nrows*no_of_samples);
	cudaMemcpy(o.data(), dev_input_delta.begin(), sizeof(float)*o.size(), cudaMemcpyDeviceToHost);
	for(const auto out : o){
		std::cout << out << ' ';
	}
	std::cout << "\n\n";

	//propogating the delta and updating weights
	int count = 0;
	for(int i=layers.size()-1; i > -1; --i){
		if(count%2==0){
			//while propagating, we need the current layer and the activation derivative of the previous layer
			propagate(layers[i], dev_input_delta, dev_output_delta, layer_outputs[i], no_of_samples);	
		}
		else{
			propagate(layers[i], dev_output_delta, dev_input_delta, layer_outputs[i], no_of_samples);	
		}
		count++;

		//debug
		basic_matrix<float> output(no_of_samples, layers[i].ncols);
		if(count%2 == 0){
			auto result = cudaMemcpy(output.data(), dev_input_delta.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
			if(result != cudaSuccess){
				throw std::runtime_error("failed to copy to host!");
			}
		}
		else{
			auto result = cudaMemcpy(output.data(), dev_output_delta.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
			if(result != cudaSuccess){
				std::cout << cudaGetErrorString(result) << '\n';
				throw std::runtime_error("failed to copy to host!");
			}
		}
		std::cout << "DELTA\n";
		output.show();	
		std::cout << '\n';

	}

}


void Network::print_weights(std::ostream &out){
/*
 * prints the weights and biases
 * first the weights are printed
 * followed by the bias leaving a line in between
 */
	for(auto layer : layers){
		layer.show();
		std::cout << '\n';
		layer.show_bias();
		std::cout << "*******\n";
	}
}

