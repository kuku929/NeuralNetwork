#include "NeuralNetwork.h"
#include "globals.h"
#include <chrono>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath> //pow
using namespace std;
std::ofstream dout("debug.txt");
using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1>>;
 
int main(int argc, char **argv){
	std::chrono::time_point<Clock> m_beg = Clock::now();
	if(argc < 3){
		cout << "enter the training size and verbosity!\n";
		return -1;
	}

	vector<string> allowed_activation_func = {"sigmoid", "Linear", "ReLU", "Leaky"}; 
	try{
		//checking if layer sizes are within bounds
		for(const int layer_size : LAYER_SIZES){
			if(layer_size < 1)
				throw invalid_argument("layer size cannot be non-positive!");
		}

		//checking for invalid activation function
		for(const string function : ACTIVATION_FUNCTIONS){
			if(find(allowed_activation_func.begin(), allowed_activation_func.end(),function) == allowed_activation_func.end())
				throw invalid_argument("invalid activation function! the following are supported:\n- sigmoid\n- Linear\n- ReLU\n- Leaky");
		}
	}
	catch(invalid_argument &e){
		cerr << e.what() << '\n';
		return -1;
	}

	Network net(LAYER_SIZES, ACTIVATION_FUNCTIONS);
	vector<float> input({1.0f,1.0f});
	
	vector<vector<float>> v = {{1,0}, {1,1},{0,1},{0,0}};
	vector<float> a = {1,0,1,0};
	int size = atoi(argv[1]);
	int verbose = atoi(argv[2]);

	//initial weights and bias
	if(verbose){
		cout << "**********************\nINITIAL\n";
		net.print_weights();
	}

	vector<int> count(4,0);
	int index;
	for(int i=0;i<size;i++){
		index = rand()%4;
		count[index]++;	
		if(verbose==2){
			cout << "INPUT: "<< v[index][0]<< ' '<<v[index][1]<< endl;
			cout << "PREDICTION Before: " <<endl;
			for(auto t : v){
				vector<float> ans = net.predict(t);
				cout << t[0]<< ' ' << t[1]<<' ';
				cout << "PREDICTION: "<<ans[0]<<'\n';
			}
		}

		//net.predict(input);
		net.fit(v[index], vector<float>({a[index]}), INITIAL_LEARNING_RATE, GRAD_DECAY, max(verbose, 1)-1, OPTIMIZER); //if verbose is 2 only then show these steps


		if(verbose==2){
			cout << "PREDICTION After: " <<endl;
			for(auto t : v){
				vector<float> ans = net.predict(t);
				cout << t[0]<< ' ' << t[1]<<' ';
				cout << "PREDICTION: "<<ans[0]<<'\n';
			}
			cout << "********\n";
		}
	}

	//count
	if(verbose){
		std::cout << "time taken: "<<std::chrono::duration_cast<Second>(Clock::now() - m_beg).count()<<'\n'; 
		cout << "**********************\n";
		cout << "COUNT: ";
		for(auto t: count) cout << t<< ' ';
		cout << "\n**********************\n";
	}
	
	//final_predictions
	float cost = 0.0f;
	for(int i=0;i<4;++i){
		vector<float> ans = net.predict(v[i]);
		cost += (a[i]-ans[0])*(a[i]-ans[0])/2.0f;
		if(verbose){
			cout << v[i][0]<< ' ' << v[i][1]<<' ';
			cout << "PREDICTION: "<<ans[0]<<'\n';
		}

	}
	if(verbose == 0){
		cout << cost << '\n';
	}

	//final weights and bias
	if(verbose){
		cout << "**********************\nFINAL\n";
		net.print_weights();
	}
	
}
