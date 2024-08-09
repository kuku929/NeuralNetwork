#include "activation.h"
#include "dev_vector.h"
// #include "kernel.h"
using namespace activation;

template<typename f_func_ptr>
__global__ void f_activate(float *input, float *output, int N, int M, f_func_ptr activ_func){
	/*
	* @brief : templated forward activation function kernel for forward pass
	* @note :
	* input is NxM
	* M is expected to be no of samples
	* each sample will be evaluated by a block
	* thus, threadId corresponds to row
	*/ 
	int row_in_matrix = threadIdx.x;  // row corresponds to node
	int col_in_matrix = blockIdx.x; // col corresponds to sample
	int index = row_in_matrix*M + col_in_matrix;
	if(row_in_matrix < N && col_in_matrix < M){

		// //debug
		// if(index == 2)
		// 	printf("testing : %f\n", input[0]);

		output[index] = activ_func(input[index]);
	}
}

template<typename b_func_ptr>
__global__ void b_activate(float *input, float *layer_output, float *output, int N, int M, b_func_ptr activ_func){
	/*
	* @brief : templated backward activation function kernel for back pass
	* @note :
	* each sample will be evaluated by a block
	* thus, threadId corresponds to column 
	* input is NxM
	* layer output is MxN
	*/ 
	int row_in_matrix = blockIdx.x;  // row should correspond to the sample
	int col_in_matrix = threadIdx.x; // col should correspond to the node
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

void Linear::forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples){
	//todo : make this faster, redundant copy
	if(output.data() == input.data())
		return;
	output = input;
}

void Linear::back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples){
	/*
	* @note :
	* input is (no of samples, size)
	* layer_output is (size, no of samples)
	* output is (no of samples, size)
	*/
	if(output.data() == input.data())
		return;
	output = input;
}

void Sigmoid::forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples){
	/*
	* @note :
	* input is (size, no of samples)
	*/
	dim3 dim_block(32, 32);
	dim3 dim_grid((size + dim_block.x - 1)/dim_block.x, (no_of_samples + dim_block.y - 1)/dim_block.y);
	f_activate<<<dim_grid, dim_block>>>(input.data(), output.data(), size, no_of_samples, f_sigmoid_);
	cudaDeviceSynchronize();
}

void Sigmoid::back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples){
	/*
	* @note :
	* input is (no of samples, size)
	* layer_output is (size, no of samples)
	* output is (no of samples, size)
	*/
	dim3 dim_block(32, 32);
	dim3 dim_grid((no_of_samples + dim_block.x - 1)/dim_block.x, (size + dim_block.y - 1)/dim_block.y);
	b_activate<<<dim_grid, dim_block>>>(input.data(), layer_output.data(), output.data(), no_of_samples, size, b_sigmoid_);
	cudaDeviceSynchronize();
}

void ReLU::forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples){
	/*
	* @note :
	* input is (size, no of samples)
	*/
	dim3 dim_block(size, 1);
	dim3 dim_grid(no_of_samples, 1);
	f_activate<<<dim_grid, dim_block>>>(input.data(), output.data(), size, no_of_samples, f_relu_);
	cudaDeviceSynchronize();
}

void ReLU::back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples){
	/*
	* @note :
	* input is (no of samples, size)
	* layer_output is (size, no of samples)
	* output is (no of samples, size)
	*/
	dim3 dim_block(size, 1); // a block will evaluate one sample
	dim3 dim_grid(no_of_samples, 1); // one block per sample
	b_activate<<<dim_grid, dim_block>>>(input.data(), layer_output.data(), output.data(), no_of_samples, size, b_relu_);
	cudaDeviceSynchronize();
}


