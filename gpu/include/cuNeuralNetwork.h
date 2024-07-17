#ifndef NEURALNET_H
#define NEURALNET_H
#include "layer.h"
#include "basic_matrix.h"
#include "activation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath> //pow

namespace net{
class Network{
	public:
		std::vector<layer::Layer> layers;
		std::vector<activation::ActivationLayer *> activation_layers;

		Network(){};
		void add_layer(size_t front_layer_size, size_t back_layer_size);
		/*
		 * adds a layer to the network
		 * input : size of the layer, activation function for that layer
		 * returns : nothing
		 */

		void add_activation(activation::ActivationLayer &activation_function);

		void print_weights(std::ostream &out=std::cout);
		/*
		 * prints weights in an orderly manner
		 * input : buffer to print to
		 * returns : nothing
		 */

		basic_matrix<float> forward_pass(basic_matrix<float> &&input);
		basic_matrix<float> forward_pass(basic_matrix<float> &input);
		/*
		 * one forward pass through the layers 
		 * input : input data to the neural network
		 * returns : output of the network as a vector
		 */
		void backward_pass(basic_matrix<float> &input, basic_matrix<float> &true_output);
		/*
		* one backward pass through the layers 
		* input : input data to the neural network, true output for the input given
		* returns : nothing
		*/	
		
		//idk why i have this
		//void train_batch(Matrix<float> inputs);
		//Network(vector<size_t> &l, vector<string> &activ_func, Matrix &w,Matrix &b); 
		//Network& operator=(const Network&) = default;
		
		//vector<float> predict(const vector<float> &input);
		//void fit(const vector<float> &input, const vector<float> &true_output, const float learning_rate, const float grad_decay, int verbose, int optimize);
		//void print_weights(ostream &out=cout);
	private:
		size_t max_layer_size=0; 
};
}
#endif
