#include "softmax.h"
using namespace nnet;

__global__ void softmax_(float *input, float *output, size_t no_of_samples, size_t size)
{
	int row = threadIdx.x;
	int col = blockIdx.x; 
	double sum=0;
	int index_in_matrix = row*no_of_samples + col;
	// note : each thread calculates this, make it shared?
	// iterate through the column
	for(int i = col; i < size*no_of_samples + col; i+=no_of_samples){
		// things can overflow, idk the fix
		sum += exp(input[i]);
	}
	output[index_in_matrix] = exp(input[index_in_matrix])/sum;
}

void Softmax::forward_pass(const dev_vector<float> &input, dev_vector<float> &output, size_t no_of_samples)
{
	dim3 dim_block(dim.second, 1);
	dim3 dim_grid(no_of_samples, 1);
	softmax_<<<dim_grid, dim_block>>>(input.data(), output.data(), no_of_samples, dim.second);
}

void Softmax::back_pass(const dev_vector<float> &input, dev_vector<float> &output, size_t no_of_samples){}
