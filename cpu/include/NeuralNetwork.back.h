#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath> //pow
#include <map>
using namespace std;
extern ofstream dout;
extern ofstream plt;
extern float completion_reward;
typedef float (*activation_funcptr)(float);

struct Network{
	vector<vector<float>> weights;
	vector<vector<float>> bias;
	vector<vector<float>> gradient_sum;
	vector<vector<float>> bias_grad_sum;
	vector<int> layer_sizes;
	map<string, activation_funcptr> activation_func_map;
	vector<string> activation_func;

	Network(){};
	Network(vector<int> &l, vector<string> &activ_func);
	Network(vector<int> &l, vector<string> &activ_func, vector<vector<float>> &w,vector<vector<float>> &b); 
	Network& operator=(const Network&) = default;
	
	vector<float> predict(const vector<float> &input);

	void fit(const vector<float> &input, const vector<float> &true_output, const float learning_rate, const float grad_decay, int verbose, int optimize);

};

float sigmoid(float weighted_sum);
float ReLU(float weighted_sum);
float Leaky(float weighted_sum);
float Linear(float weighted_sum);
float Fdash(float output, string activation_function); 
void beautiful_print(const vector<float> &input, vector<float> &hidden_weighted_sum, vector<float> &hidden_output, vector<float> &output_weighted_sum, vector<float> &output,vector<vector<float>> weights, vector<vector<float>> bias);
