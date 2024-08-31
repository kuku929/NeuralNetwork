#include "cuNeuralNetwork.h"
#include "activation.h"
#include "basic_matrix.h"
#include "optimizer.h"
#include "data_reader.h"
#include <iostream>
#include <vector>
int main(){
	//testing diabetes 
	
	// getting data
	nnet::DataReader reader("../data/diabetes.csv");
	reader.tokenize_data();
	auto data_mat = reader.convert_to_matrix(10, reader.ncols());
	auto input = reader.convert_to_matrix(10, reader.ncols()-1);
	auto output = data_mat.get_col(data_mat.ncols - 1);
	input.min_max_normalize();
	input.transpose();
	output.transpose();

	float learning_rate = 0.01f;
	float beta = 0.9f;
	uint epochs = 1000;
	nnet::Network net;
	//layer 1
	net.add_layer(input.nrows, 64);
	//activation 1
	auto activ = nnet::Sigmoid(64);
	net.add_activation(activ);
	//optimizer 1
	auto optim = nnet::RMSProp(net.layers.back(), learning_rate, beta);
	net.add_optimizer(optim);

	//layer 2
	net.add_layer(64, 64);
	//activation 2
	auto activ2 = nnet::ReLU(64);
	net.add_activation(activ2);
	//optimizer 2
	auto optim2 = nnet::RMSProp(net.layers.back(), learning_rate, beta);
	net.add_optimizer(optim2);

	//layer 3
	net.add_layer(64, output.nrows);
	//activation 2
	auto activ3 = nnet::Linear(output.nrows);
	net.add_activation(activ3);
	//optimizer 2
	auto optim3 = nnet::RMSProp(net.layers.back(), learning_rate, beta);
	net.add_optimizer(optim3);

	auto loss = nnet::MSELoss(output.nrows);
	net.add_loss(loss);
	for(int i=0;i < epochs; ++i){
		net.backward_pass(input, output);
	}
	auto o = net.forward_pass(input);
	o.show();
}
