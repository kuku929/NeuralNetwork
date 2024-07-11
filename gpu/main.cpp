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
	//auto m_beg = Clock::now();
	//std::vector<size_t> layer_sizes({2,3,1});
	//std::vector<std::string> activation_functions({"ReLU","ReLU","Linear"});
	//Network net(layer_sizes, activation_functions);
	net::Network net;
	net.add_layer(2, 1);
	auto activ = activation::Sigmoid(1);
	net.add_activation(activ);
	net.print_weights();

	//testing xor

	//std::ifstream fin("../data/data.txt");
	//size_t n = 1024;
	//int training_size = 10000;
	//basic_matrix<float> v(2,n);
	//basic_matrix<float> output(1, n);
	//float v1,v2,a;
	//for(int i=0; i < v.ncols; ++i){
		//fin >> v1 >> v2 >> a;
		//v.get(0,i) = v1;
		//v.get(1,i) = v2;
		//output.get(0,i) = a;
	//}
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

	//std::cout << "----------------------------\n";

	////basic_matrix<float> input(2,1);
	//basic_matrix<float> op(1,4);
	//op.get(0,0) = 0;
	//op.get(0,1) = 1;
	//op.get(0,2) = 1;
	//op.get(0,3) = 0;
	//net.backward_pass(test, op);

	//net.print_weights();
	//for(int i=0; i < training_size; ++i){
		//int rand_index = rand()%n;
		//input.get(0,0) = v.get(0, rand_index);
		//input.get(1,0) = v.get(1, rand_index);
		//input.get(2,0) = v.get(2, rand_index);
		//op.get(0,0) = output.get(0, rand_index);
		//net.backward_pass(input, op);
		////net.print_weights();
		//std::cout << "----done----\n";
	//}
	////output.show();
	//net.print_weights();
	//std::cout << '\n';
	//o = net.forward_pass(test);
	//o.show();
}
