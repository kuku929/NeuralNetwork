#include "activation.h"
#include "dev_vector.h"
// #include "kernel.h"
using namespace activation;
const int BLOCK_SIZE=8;


//kernel for activation function
template<typename f_func_ptr>
__global__ void f_activate(float *input, float *output, int N, int M, f_func_ptr activ_func){
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	int index = row_in_matrix*N + col_in_matrix;
	if(row_in_matrix < N && col_in_matrix < M){

		////debug
		// if(index == 0)
			// printf("testing : %f\n", activ_func(1.0f));

		output[index] = activ_func(input[index]);
	}
}

template<typename b_func_ptr>
__global__ void b_activate(float *input, float *layer_output, float *output, int N, int M, b_func_ptr activ_func){
	/*
	* input is NxM
	*/ 
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	int index = row_in_matrix*M + col_in_matrix;
	int t_index = col_in_matrix*N + row_in_matrix;
	if(row_in_matrix < N && col_in_matrix < M){
		output[index] = activ_func(input[index], layer_output[t_index]);

		// //debug
		// if(index == 1){
		// 	printf("back_actviate : %f\n", layer_output[t_index]);
		// }

	}

	
	// //debug
	// if(index == 0){
	// 	printf("testing : %f\n", output[t_index]);
	// }

}

void Linear::forward_activate(dev_vector<float> &input, dev_vector<float> &output){
	//todo : make this faster, redundant copy
	if(output.data() == input.data())
		return;
	output = input;
}

void Linear::back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output){
	if(output.data() == input.data())
		return;
	output = input;
}

void Sigmoid::forward_activate(dev_vector<float> &input, dev_vector<float> &output){
	dim3 dim_block(32, 32);
	int ncols = input.size()/size;
	dim3 dim_grid((size + dim_block.x - 1)/dim_block.x, (ncols + dim_block.y - 1)/dim_block.y);
	f_activate<<<dim_grid, dim_block>>>(input.data(), output.data(), size, ncols, f_sigmoid_);
}

void Sigmoid::back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output){
	dim3 dim_block(32, 32);
	int nrows = input.size()/size;
	dim3 dim_grid((nrows + dim_block.x - 1)/dim_block.x, (size + dim_block.y - 1)/dim_block.y);
	b_activate<<<dim_grid, dim_block>>>(input.data(), layer_output.data(), output.data(), nrows, size, b_sigmoid_);
}

void ReLU::forward_activate(dev_vector<float> &input, dev_vector<float> &output){
	dim3 dim_block(32, 32);
	int ncols = input.size()/size;
	dim3 dim_grid((size + dim_block.x - 1)/dim_block.x, (ncols + dim_block.y - 1)/dim_block.y);
	f_activate<<<dim_grid, dim_block>>>(input.data(), output.data(), size, ncols, f_relu_);
}

void ReLU::back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output){
	dim3 dim_block(32, 32);
	int nrows = input.size()/size;
	dim3 dim_grid((nrows + dim_block.x - 1)/dim_block.x, (size + dim_block.y - 1)/dim_block.y);
	b_activate<<<dim_grid, dim_block>>>(input.data(), layer_output.data(), output.data(), nrows, size, b_relu_);
}


