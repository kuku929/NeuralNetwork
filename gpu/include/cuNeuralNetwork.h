#ifndef NEURALNET_H
#define NEURALNET_H
#include "layer.h"
#include "basic_matrix.h"
#include "activation.h"
#include "optimizer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath> //pow

namespace net{
class Network{
	public:
		// right now you need an optimizer for each layer, maybe there is a better way?
		// in the future, the back_pass of the network will be written by user in main 
		// std::vector<layer::Layer> layers;
		//todo : convert to this:
		std::vector<std::shared_ptr<layer::Layer>> layers; //optimizer will also own the memory
		std::vector<activation::ActivationLayer *> activation_layers;
		std::vector<optimizer::Optimizer *> optimizers;

		Network(){};
		void add_layer(size_t front_layer_size, size_t back_layer_size);
		/*
		 * adds a layer to the network
		 * input : size of the layer, activation function for that layer
		 * returns : nothing
		 */

		void add_activation(activation::ActivationLayer &activation_function);
		void add_optimizer(optimizer::Optimizer &optimizer);

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
