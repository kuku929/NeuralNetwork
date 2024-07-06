#include "layer.h"
#include "activation.h"
#include "cuNeuralNetwork.h"
#include "dev_vector.h"
#include "kernel.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath> //pow
using namespace net;


void Network::add_layer(size_t back_layer_size, size_t front_layer_size){
 /*
  * @arguments : size of the front set and back set, activation function for the next layer
  * 
  * @brief : 
  * max_layer_size is required for later use in code while allocating 
  * device memory.
  */
	//if layers have already been added
	//max layer size is zero when object initialized
	if(front_layer_size == 0 || back_layer_size == 0){
		throw std::invalid_argument("layer cannot have 0 neurons");
	}

	layers.push_back(layer::Layer(front_layer_size, back_layer_size));

	if(front_layer_size > max_layer_size)
		max_layer_size = front_layer_size;

	if(back_layer_size > max_layer_size)
		max_layer_size = back_layer_size;
	return;
}


void Network::add_activation(activation::ActivationLayer &activation_function){
	activation_layers.push_back(&activation_function);
	return;
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

	basic_matrix<float> output(layers.back().nrows, no_of_samples);

	//copying to device
	dev_vector<float> dev_input((max_layer_size)*no_of_samples);	
	dev_input.set(input.data(), input.size);
	dev_vector<float> dev_output((max_layer_size)*no_of_samples);	

	
	//forward pass
	for(int i=0; i < layers.size(); ++i){
		layers[i].forward_pass(dev_input, dev_output, no_of_samples);
		activation_layers[i]->forward_activate(dev_output, dev_input);
	}

	//copying to host
	auto result = cudaMemcpy(output.data(), dev_input.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
	if(result != cudaSuccess){
		throw std::runtime_error("failed to copy to host!");
	}

	return output;

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

	////debug
	//std::vector<basic_matrix<float>> h_layer_outputs(layers.size()+1);
	//h_layer_outputs[0] = input;

	//adding the initial input
	layer_outputs[0] = std::make_shared<dev_vector<float>>(input);	
	//for the hidden layers
	for(int j=0; j < layers.size(); ++j){
		layer_outputs[j+1] = std::make_shared<dev_vector<float>>((layers[j].nrows)*no_of_samples);

		////debug
		//h_layer_outputs[j+1] = basic_matrix<float>(layers[j].nrows, no_of_samples);

	}

	//forward pass
	for(int i=0; i < layers.size(); ++i){
		layers[i].forward_pass(*layer_outputs[i], *layer_outputs[i+1], no_of_samples);
		//will this work?
		activation_layers[i]->forward_activate(*layer_outputs[i+1], *layer_outputs[i+1]);
	}
	
	////debug
	//for(int i=0; i < layer_outputs.size(); ++i){
		//cudaMemcpy(h_layer_outputs[i].data(), layer_outputs[i]->data(), sizeof(float)*layer_outputs[i]->size(), cudaMemcpyDeviceToHost); 
	//}
	//std::cout << "LAYER OUTPUTS\n";
	//for(auto const a : h_layer_outputs){
		//a.show();
		//std::cout << '\n';
	//}

	//finding intial delta
	dev_vector<float> dev_true_output(true_output);

	//allocating maximum possible data to prevent needless copying
	dev_vector<float> dev_input_delta(no_of_samples*max_layer_size);
	dev_vector<float> dev_output_delta(no_of_samples*max_layer_size);


	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((true_output.ncols + dim_block.x - 1)/dim_block.x, (true_output.nrows + dim_block.y - 1)/dim_block.y);

	//finding the difference and transposing the resulting matrix
	//delta will have no_of_samples rows and layer size cols
	find_delta_and_transpose<<<dim_grid, dim_block>>>(dev_true_output.data(), (*layer_outputs.back()).data(), dev_input_delta.data(), true_output.nrows, true_output.ncols);

	////debug
	//std::cout << "DELTA\n";
	//std::vector<float> o(layers.back().nrows*no_of_samples);
	//cudaMemcpy(o.data(), dev_input_delta.begin(), sizeof(float)*o.size(), cudaMemcpyDeviceToHost);
	//for(const auto out : o){
		//std::cout << out << ' ';
	//}
	//std::cout << "\n\n";

	//propogating the delta and updating weights
	for(int i=layers.size()-1; i > -1; --i){
		//while propagating, we need the current layer and the activation derivative of the previous layer
		layers[i].back_pass(dev_input_delta, dev_output_delta, layer_outputs[i], no_of_samples);	
		activation_layers[i]->back_activate(dev_output_delta, dev_input_delta);

		////debug
		//basic_matrix<float> output(no_of_samples, layers[i].ncols);
		//if(count%2 == 0){
			//auto result = cudaMemcpy(output.data(), dev_input_delta.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
			//if(result != cudaSuccess){
				//throw std::runtime_error("failed to copy to host!");
			//}
		//}
		//else{
			//auto result = cudaMemcpy(output.data(), dev_output_delta.begin(), sizeof(float)*output.size, cudaMemcpyDeviceToHost);
			//if(result != cudaSuccess){
				//std::cout << cudaGetErrorString(result) << '\n';
				//throw std::runtime_error("failed to copy to host!");
			//}
		//}
		//std::cout << "DELTA\n";
		//output.show();	
		//std::cout << '\n';

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

