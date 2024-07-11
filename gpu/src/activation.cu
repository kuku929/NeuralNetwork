#include "activation.h"
#include "dev_vector.h"
// #include "kernel.h"
using namespace activation;
const int BLOCK_SIZE=8;

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

//kernel for activation function
template<typename f_func_ptr>
__global__ void f_activate(float *a, float *b, int N, int M, f_func_ptr activ_func){
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	int index = row_in_matrix*N + col_in_matrix;
	if(row_in_matrix < N && col_in_matrix < M){

		////debug
		// if(index == 0)
			// printf("testing : %f\n", activ_func(1.0f));

		b[index] = activ_func(a[index]);
	}
}

template<typename b_func_ptr>
__global__ void b_activate(float *a, float *b, float *c, int N, int M, b_func_ptr activ_func){
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	int index = row_in_matrix*N + col_in_matrix;
	int t_index = col_in_matrix*M + row_in_matrix;
	if(row_in_matrix < N && col_in_matrix < M){
		b[index] = activ_func(a[index], c[t_index]);
	}
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


