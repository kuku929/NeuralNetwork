#include "cuNeuralNetwork.h"
#include "activation.h"
#include "basic_matrix.h"
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
	net::Network net;
	net.add_layer(2, 4);
	auto activ = activation::ReLU(4);
	net.add_activation(activ);
	net.add_layer(4, 1);
	auto activ2 = activation::Linear(1);
	net.add_activation(activ2);
	net.print_weights();

	basic_matrix<float> test(2,4);
	test.get(0,0) = 1;
	test.get(1,0) = 1;

	test.get(0,1) = 0;
	test.get(1,1) = 1;


	test.get(0,2) = 1;
	test.get(1,2) = 0;

	test.get(0,3) = 0;
	test.get(1,3) = 0;

	basic_matrix<float> o = net.forward_pass(test);
	o.show();

	std::cout << "----------------------------\n";

	basic_matrix<float> op(1,4);
	op.get(0,0) = 0;
	op.get(0,1) = 1;
	op.get(0,2) = 1;
	op.get(0,3) = 0;
	for(int i=0; i < 1000; ++i){
		net.backward_pass(test, op);
	}
	o = net.forward_pass(test);
	o.show();

	std::cout << "final weights :\n";
	net.print_weights();

}
