#pragma once
#include <stdio.h>
const int BLOCK_SIZE =1;
typedef float (*funcptr)(float);


__global__ void find_delta_and_transpose(float *a, float *b, float *output, int N, int M, const funcptr function){
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	int index_in_vector = row_in_matrix*M + col_in_matrix;
	int transpose_index_in_vector = col_in_matrix*N + row_in_matrix;
	if(row_in_matrix < N && col_in_matrix< M){
		output[transpose_index_in_vector] = (a[index_in_vector] - b[index_in_vector])*function(b[index_in_vector]);
	}
}

__global__ void dev_vec_matmul(const float *dev_a, const float *dev_b, float *dev_output, int N, int M, const funcptr function){
	//a is NxM
	int ROW = threadIdx.x;
	float temp_value=0;

	//if(ROW == 0){
		//printf("%f\n", activation_func_map[activ_func_ind](-1.0f));
	//}
	for(size_t i=0; i < M; ++i){
		temp_value += dev_a[ROW*M + i]*dev_b[i];
	}
	dev_output[ROW] = function(temp_value);
}

__global__ void gemm_function(const float *a, const float *b, const float *c, float *output, int N, int M, int K, const funcptr function){
	//a is NxK
	//b is KxM
	//c is Nx1
	//performs the following general operation : return f(a*b + c), where f() is applied to each element
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	//loop to find the sub-matrix of output
	//iterates through sub-matrices of A and B to copy to shared memory
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	if(row_in_matrix < N && col_in_matrix< M){
		float temp_value=0;
		for(int i=0; i < K; i+=BLOCK_SIZE){
			__shared__ float A[BLOCK_SIZE][BLOCK_SIZE], B[BLOCK_SIZE][BLOCK_SIZE]; 

			//copying to shared memory
			for(int j=0; j < BLOCK_SIZE; ++j){
				for(int k=0; k < BLOCK_SIZE; ++k){
					A[j][k] = a[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					B[j][k] = b[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
				}
			}


			//wait for completion
			__syncthreads();

			//multiply and add
			for(int j=0; j < BLOCK_SIZE; ++j){
				temp_value += A[row][j]*B[j][col];
			}

			__syncthreads();

		}

		output[(row_in_matrix)*M + col_in_matrix] = function(temp_value+c[row_in_matrix]);
	}
}

__global__ void matmul_funcmul(const float *a, const float *b, float *c, float *output, int N, int M, int K, const funcptr function){
	//a is NxK
	//b is KxM
	//c is NxM
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	//loop to find the sub-matrix of output
	//iterates through sub-matrices of A and B to copy to shared memory
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	if(row_in_matrix < N && col_in_matrix< M){
		float temp_value=0;
		for(int i=0; i < K; i+=BLOCK_SIZE){
			__shared__ float A[BLOCK_SIZE][BLOCK_SIZE], B[BLOCK_SIZE][BLOCK_SIZE]; 

			//copying to shared memory
			for(int j=0; j < BLOCK_SIZE; ++j){
				for(int k=0; k < BLOCK_SIZE; ++k){
					A[j][k] = a[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					B[j][k] = b[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
				}
			}

			//if(row == 0 && col == 0 && bx == 0 && by == 0){
				//printf("kernel A : %f\n", A[0][0]);
				//printf("kernel B : %f\n", B[0][0]);
			//}


			//wait for completion
			__syncthreads();

			//multiply and add
			for(int j=0; j < BLOCK_SIZE; ++j){
				temp_value += A[row][j]*B[j][col];
			}

			__syncthreads();


		}
		//debug
		//if(row == 0 && col == 0 && bx == 0 && by == 0){
			//printf("kernel : %f\n", (*function)(-0.5));
		//}

		int index = (row_in_matrix)*M + col_in_matrix; 
		int transposed_index =(col_in_matrix)*N + row_in_matrix;
		output[index] = temp_value*function(c[transposed_index]);
	}
}

__global__ void update_bias(float *dev_bias, float *layer_delta, int N, int M, float learning_rate){ 
	int COL = threadIdx.x;
	float temp_value=0;

	//if(ROW == 0){
		//printf("%f\n", activation_func_map[activ_func_ind](-1.0f));
	//}
	for(size_t i=0; i < N; ++i){
		temp_value += layer_delta[i*M + COL];
	}
	dev_bias[COL] += learning_rate*temp_value/N;
}
__global__ void update_weights(float *dev_weights, float *layer_output, float *layer_delta, int N, int M, int K, float learning_rate){
/*
 * will multiply layer_output and layer_delta together
 * transpose(weights) += layer_output*layer_delta*learning_rate
 *
 * layer_outputs is NxK
 * layer_deltas is KxM
 * K is no of samples
 * 
 * assume a layer is (4,2). i.e. 4 rows and 2 cols
 * updating the weights of this layer requires the following matrices
 * layer_output ---> (2,no_of_samples)
 * layer_deltas ---> (no_of_samples, 4)
 * 
 * doing layer_output*layer_deltas gives a (2,4) matrix. 
 * thus we need to transpose it.
 */
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	if(row_in_matrix < N && col_in_matrix< M){
		float temp_value = 0;
		for(int i=0; i < K; i+=BLOCK_SIZE){
			__shared__ float A[BLOCK_SIZE][BLOCK_SIZE], B[BLOCK_SIZE][BLOCK_SIZE]; 

			//copying to shared memory
			for(int j=0; j < BLOCK_SIZE; ++j){
				for(int k=0; k < BLOCK_SIZE; ++k){
					A[j][k] = layer_output[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 

					B[j][k] = layer_delta[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
				}
			}

			//wait for completion
			__syncthreads();

			//multiply and add
			for(int j=0; j < BLOCK_SIZE; ++j){
				temp_value += A[row][j]*B[j][col];
			}

			__syncthreads();
		}

		//transpose added
		//debug
		//if(row == 0 && col == 0 && bx == 0 && by == 0){
			//printf("%f\n", A[0][0]);
		//}
		//printf("%f\n\n", learning_rate*temp_value);
		dev_weights[row_in_matrix*M + col_in_matrix] += learning_rate*temp_value/K;
	}
}

