#pragma once
#include "basic_matrix.h"
#include <iostream>
#include <vector>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
typedef float (*activation_funcptr)(float);
class Layer : public basic_matrix<float>{
	/*
	 * this class inherits from basic_matrix
	 * contains weights and bias implemented as 1-D vectors
	 */
	public:
		activation_funcptr activation_function;
		activation_funcptr back_activation_derivative;
		activation_funcptr activation_derivative;
		activation_funcptr null_function;
		Layer(): basic_matrix(){};
		Layer(int N, int M, int activ_func_id, Layer &prev_layer);
		Layer(int N, int M, int activ_func_id);
		
		void show_bias(){
			for(auto b : bias_){
				std::cout << b << ' ';
			}
			std::cout << '\n';
		}

		float* bias(){
			/*
			 * returns pointer to the first element of the vector 
			 * used while copying data from device memory to host
			 */
			return &bias_[0];
		}

		const float* bias() const{
			/*
			 * returns pointer to the first element of the vector 
			 * used when the matrix is not being updated 
			 */
			return bias_.data();
		}

		const float& get_bias(int row)const{
			/*
			 * returns (i,j)th element in the matrix
			 * used when value does not need to be changed
			 */
			return bias_[row];
		}


		float& get_bias(int row){
			/*
			 * returns (i,j)th element in the matrix
			 * used when value needs to be changed
			 */
			return bias_[row];
		}
		const float& get_weights(int row, int col)const{
			/*
			 * returns (i,j)th element in the matrix
			 * used when value does not need to be changed
			 */
			return data_[row*ncols + col];
		}


		float& get_weights(int row, int col){
			/*
			 * returns (i,j)th element in the matrix
			 * used when value needs to be changed
			 */
			return data_[row*ncols + col];
		}

	private:
		std::vector<float> bias_;

};
