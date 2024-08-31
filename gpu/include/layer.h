#ifndef LAYER_H
#define LAYER_H

#include "basic_matrix.h"
#include "dev_vector.h"
#include "dimension.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cstdio>
namespace nnet{
class Optimizer;
class RMSProp;

class Layer : public basic_matrix<float>{
	/*
	 * this class inherits from basic_matrix
	 * contains weights and bias implemented as 1-D vectors
	 */
	public:
		Layer(): basic_matrix(), dim(){};
		Layer(int N, int M);
		~Layer() = default;
		void forward_pass(const dev_vector<float> &input, dev_vector<float> &output, const size_t no_of_samples);
		void back_pass(const dev_vector<float> &input, dev_vector<float> &output, const size_t no_of_samples);
		void update(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples);
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

		void copy_to_host();

		void show_dev(){
			std::cout << "show:\n";
			cudaMemcpy(this->data(), dev_weights_->data(), sizeof(float)*dev_weights_->size(), cudaMemcpyDeviceToHost);
			this->show();
		}

	private:
		Dimension dim;
		std::vector<float> bias_;
		std::shared_ptr<dev_vector<float>> dev_weights_;
		std::shared_ptr<dev_vector<float>> dev_bias_;
	friend class nnet::RMSProp;
};
}
#endif //LAYER_H
