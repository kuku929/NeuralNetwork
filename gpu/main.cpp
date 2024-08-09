#include "cuNeuralNetwork.h"
#include "activation.h"
#include "basic_matrix.h"
#include "optimizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>
#include <iomanip>
using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1>>;
int main(){
	//testing xor
	float learning_rate = 0.09f;
	float beta = 0.9f;
	uint epochs = 1000;
	net::Network net;
	//layer 1
	net.add_layer(2, 4);
	// std::cout << "testing layer -1 : " << net.layers.back()->nrows << ' ' << net.layers.back()->ncols << '\n';
	//activation 1
	auto activ = activation::ReLU(4);
	net.add_activation(activ);
	//optimizer 1
	auto optim = optimizer::RMSProp(net.layers.back(), learning_rate, beta);
	net.add_optimizer(optim);

	//layer 2
	net.add_layer(4, 1);
	//activation 2
	auto activ2 = activation::Linear(1);
	net.add_activation(activ2);
	//optimizer 2
	auto optim2 = optimizer::RMSProp(net.layers.back(), learning_rate, beta);
	net.add_optimizer(optim2);

	net.print_weights();

	basic_matrix<float> test(2,4);
	test.get(0,0) = 1;
	test.get(1,0) = 0;

	test.get(0,1) = 1;
	test.get(1,1) = 1;


	test.get(0,2) = 0;
	test.get(1,2) = 1;

	test.get(0,3) = 0;
	test.get(1,3) = 0;


	basic_matrix<float> o = net.forward_pass(test);
	o.show();

	std::cout << "----------------------------\n";

	basic_matrix<float> op(1,4);
	op.get(0,0) = 1;
	op.get(0,1) = 0;
	op.get(0,2) = 1;
	op.get(0,3) = 0;

	basic_matrix<float> sub_test(2,1);
	basic_matrix<float> sub_op(1,1);

	srand(1);
	for(int i=0; i < epochs; ++i){
		int ind = rand()%4;
		sub_test=test.get_col(ind);
		sub_op=op.get_col(ind);

		// std::cout << "INPUT : \n";
		// sub_test.show();

		// std::cout << "PREDICATION Before:\n";
		// o = net.forward_pass(test);
		// o.show();

		net.backward_pass(test, op);

		// std::cout << "PREDICATION After:\n";
		// o = net.forward_pass(test);
		// o.show();

	}

	std::cout << "final output :\n";
	o = net.forward_pass(test);
	o.show();

	std::cout << "final weights :\n";
	net.print_weights();

}
