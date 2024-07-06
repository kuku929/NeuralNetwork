#include "NeuralNetwork.h"
#include "globals.h"
#include <chrono>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath> //pow
std::ofstream dout("debug.txt");
using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1>>;

int main(){
	Network net(LAYER_SIZES, ACTIVATION_FUNCTIONS);

	//training data
	std::ifstream train_images("../data/mnist_train_images_filtered");
	std::ifstream train_labels("../data/mnist_train_labels_filtered");
	size_t n = 1024;
	int input_size = 784;
	std::vector<std::vector<float>> train_input(n);
	float v1,v2,a;
	for(int i=0; i < train_input.size(); ++i){
		train_input[i].resize(input_size);
		for(int j=0; j < input_size; ++j){
			train_images >> train_input[i][j];
		}
	}

	std::chrono::time_point<Clock> m_beg = Clock::now();
	for(int i=0; i < n; ++i){
		net.predict(train_input[i]);
	}
	std::cout << "time taken: "<<std::chrono::duration_cast<Second>(Clock::now() - m_beg).count()<<'\n';
}
