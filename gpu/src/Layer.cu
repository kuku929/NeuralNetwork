#include "helper_functions.h" 
#include "dev_vector.h"
#include "Layer.h"
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>
#include <cstdio>
typedef float (*activation_funcptr)(float);

Layer::Layer(int N, int M, int activ_func_id, Layer &prev_layer): basic_matrix(N,M){
	bias_.resize(N);	
	std::vector<activation_funcptr> host_activation_func_map(4);
	std::vector<activation_funcptr> host_activation_func_der_map(4);
	float range = 2.0f;

	//copying activation function to host
	auto result = cudaMemcpyFromSymbol(host_activation_func_map.data(), activation_func_map, sizeof(activation_funcptr)*host_activation_func_map.size());
	if(result != cudaSuccess){
		std::cout << cudaGetErrorString(result);
	}
	this->activation_function = host_activation_func_map[activ_func_id];

	//copying the null function to host
	result = cudaMemcpyFromSymbol(&null_function, null_funcptr, sizeof(activation_funcptr));
	if(result != cudaSuccess){
		std::cout << cudaGetErrorString(result);
	}

	//copying the derivative of activation function to host
	result = cudaMemcpyFromSymbol(host_activation_func_der_map.data(), activation_func_der_map, sizeof(activation_funcptr)*host_activation_func_der_map.size());
	if(result != cudaSuccess){
		std::cout << cudaGetErrorString(result);
	}
	
	this->activation_derivative = host_activation_func_der_map[activ_func_id];
	this->back_activation_derivative = prev_layer.activation_derivative;

	if(!data_.empty()){
		for(size_t i=0;i<nrows;++i){
			this->get_bias(i) = 0.0f; //bias for the next layer
			for(size_t j=0;j<ncols;++j){
				float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
				this->get_weights(i, j) = normalized_value*range; 
			}
		}
	}
}


Layer::Layer(int N, int M, int activ_func_id): basic_matrix(N,M){
	bias_.resize(N);	
	std::vector<activation_funcptr> host_activation_func_map(4);
	std::vector<activation_funcptr> host_activation_func_der_map(4);
	float range = 2.0f;

	//copying activation function to host
	auto result = cudaMemcpyFromSymbol(host_activation_func_map.data(), activation_func_map, sizeof(activation_funcptr)*host_activation_func_map.size());
	if(result != cudaSuccess){
		std::cout << cudaGetErrorString(result);
	}
	this->activation_function = host_activation_func_map[activ_func_id];

	//copying the null function to host
	result = cudaMemcpyFromSymbol(&null_function, null_funcptr, sizeof(activation_funcptr));
	if(result != cudaSuccess){
		std::cout << cudaGetErrorString(result);
	}

	//copying the derivative of activation function to host
	result = cudaMemcpyFromSymbol(host_activation_func_der_map.data(), activation_func_der_map, sizeof(activation_funcptr)*host_activation_func_der_map.size());
	if(result != cudaSuccess){
		std::cout << cudaGetErrorString(result);
	}
	
	//Linear back activation function for the first layer
	std::cout << "activation: " << activ_func_id << '\n';
	this->activation_derivative = host_activation_func_der_map[activ_func_id];
	this->back_activation_derivative = host_activation_func_der_map[3];

	if(!data_.empty()){
		for(size_t i=0;i<nrows;++i){
			this->get_bias(i) = 0.0f; //bias for the next layer
			for(size_t j=0;j<ncols;++j){
				float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
				this->get_weights(i, j) = normalized_value*range; 
			}
		}
	}
}
