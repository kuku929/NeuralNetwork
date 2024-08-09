#include <stdio.h>
#include "kernel.h"
const int BLOCK_SIZE=8;
typedef float (*f_func_ptr)(const float ); 
typedef float (*b_func_ptr)(const float , const float ); 

__global__ void initialize_gradient_(float *grad_bias, float *grad_weights, int M){
	//grad weights has M columns
	int row = blockIdx.x;
	int col = threadIdx.x;	

	if(col == 0){
		grad_bias[row] = 0.0f;
	}
	grad_weights[row*M + col] = 0.0f;
}

__global__ void find_delta_and_transpose(float *a, float *b, float *output, int N, int M){
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	int row_in_matrix = by*BLOCK_SIZE+row;
	int col_in_matrix = bx*BLOCK_SIZE+col;
	int index_in_vector = row_in_matrix*M + col_in_matrix;
	int transpose_index_in_vector = col_in_matrix*N + row_in_matrix;
	if(row_in_matrix < N && col_in_matrix< M){
		output[transpose_index_in_vector] = a[index_in_vector] - b[index_in_vector];

		// //debug
		// if(index_in_vector == 0){
		// 	for(int i=0;i < N; ++i)printf("find_delta: %f ", b[i]); 
		// }

	}

}

__global__ void dev_vec_matmul(const float *dev_a, const float *dev_b, float *dev_output, int N, int M){
	//a is NxM
	int ROW = threadIdx.x;
	float temp_value=0;

	//// debug
	//if(ROW == 0){
		//printf("%f\n", activation_func_map[activ_func_ind](-1.0f));
	//}

	for(size_t i=0; i < M; ++i){
		temp_value += dev_a[ROW*M + i]*dev_b[i];
	}
	dev_output[ROW] = temp_value;
}

__global__ void gemm(const float *a, const float *b, const float *c, float *output, int N, int M, int K){
	//a is NxK
	//b is KxM
	//c is Nx1
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
					// this is not good, maybe BLOCK_SIZE can change depending upon the network arch
					// should i move this if outside, like (by+1)*BLOCK_SIZE < N? maybe wrap divergence may happen then
					if(by*BLOCK_SIZE+j < N && i+k < K){
						A[j][k] = a[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					}else{
						A[j][k] = 0;
					}
					if(i+j < K && bx*BLOCK_SIZE+k < M){
						B[j][k] = b[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
					}else{
						B[j][k] = 0;
					}
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

		// //debug
		// int index = row_in_matrix*M + col_in_matrix;
		// if(index == 0){
		// 	printf("gemm : %f\n", temp_value+c[0]);
		// }

		output[(row_in_matrix)*M + col_in_matrix] = temp_value+c[row_in_matrix];
	}
}

__global__ void matmul(const float *a, const float *b, float *output, int N, int M, int K){
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
					if(by*BLOCK_SIZE+j < N && i+k < K){
						A[j][k] = a[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					}else{
						A[j][k] = 0;
					}
					if(i+j < K && bx*BLOCK_SIZE+k < M){
						B[j][k] = b[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
					}else{
						B[j][k] = 0;
					}
				}
			}

			////debug
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

		////debug
		//if(row == 0 && col == 0 && bx == 0 && by == 0){
			//printf("kernel : %f\n", (*function)(-0.5));
		//}

		//int transposed_index =(col_in_matrix)*N + row_in_matrix;
		output[(row_in_matrix)*M + col_in_matrix] = temp_value;
	}
}

__global__ void update_bias(float *dev_bias, float *layer_delta, int N, int M, float learning_rate){ 
	/*
	* bias will be (M, 1) dim
	* N is no of samples
	* M is rows of the layer 
	*/
	int COL = threadIdx.x;
	float temp_value=0;

	//// debug
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
					if(by*BLOCK_SIZE+j < N && i+k < K){
						A[j][k] = layer_output[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					}else{
						A[j][k] = 0;
					}
					if(i+j < K && bx*BLOCK_SIZE+k < M){
						B[j][k] = layer_delta[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
					}else{
						B[j][k] = 0;
					}
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

		////debug
		//if(row == 0 && col == 0 && bx == 0 && by == 0){
			//printf("%f\n", A[0][0]);
		//}
		//printf("%f\n\n", learning_rate*temp_value);

		//NOTE : both these statements work for xor at least, wtf
		// dev_weights[row_in_matrix*M + col_in_matrix] += learning_rate*temp_value/K;
		dev_weights[col_in_matrix*N + row_in_matrix] += learning_rate*temp_value/K;
	}
}

__global__ void rmsprop_update_bias_(float *dev_bias, float *layer_delta, float *gradient_sum, int N, int M, float learning_rate, float beta){ 
	// layer_delta is NxM
	int COL = threadIdx.x;
	float temp_value=0;

	for(size_t i=0; i < N; ++i){
		temp_value += layer_delta[i*M + COL];
	}

	// //debug
	// if(COL == 0){
	// 	printf("rmsprop_bias_kernel : %f\n", layer_delta[0]);
	// }

	gradient_sum[COL] = gradient_sum[COL]*beta + (1-beta)*temp_value*temp_value;
	dev_bias[COL] += learning_rate*temp_value/(1e-5 + sqrt(gradient_sum[COL]));
	// NOTE : MAX SCAMM
	// dev_bias[COL] += learning_rate*temp_value/(1 + sqrt(gradient_sum[COL]));
}

__global__ void rmsprop_update_weights_(float *dev_weights, float *layer_output, float *layer_delta, float *gradient_sum, int N, int M, int K, float learning_rate, float beta){
/*
 * will multiply layer_output and layer_delta together
 * transpose(weights) += layer_output*layer_delta*learning_rate/root(gradient_sum)
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
	// int index_in_matrix = row_in_matrix*M + col_in_matrix;
	if(row_in_matrix < N && col_in_matrix< M){
		float temp_value = 0;
		for(int i=0; i < K; i+=BLOCK_SIZE){
			__shared__ float A[BLOCK_SIZE][BLOCK_SIZE], B[BLOCK_SIZE][BLOCK_SIZE]; 

			//copying to shared memory
			for(int j=0; j < BLOCK_SIZE; ++j){
				for(int k=0; k < BLOCK_SIZE; ++k){
					if(by*BLOCK_SIZE+j < N && i+k < K){
						A[j][k] = layer_output[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					}else{
						A[j][k] = 0;
					}
					if(i+j < K && bx*BLOCK_SIZE+k < M){
						B[j][k] = layer_delta[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
					}else{
						B[j][k] = 0;
					}

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

		////debug
		//if(row == 0 && col == 0 && bx == 0 && by == 0){
			//printf("%f\n", A[0][0]);
		//}
		//printf("%f\n\n", learning_rate*temp_value);


		//should this be transposed???
		int transposed_ind = col_in_matrix*N + row_in_matrix;
		gradient_sum[transposed_ind] = beta*gradient_sum[transposed_ind] + (1-beta)*temp_value*temp_value;
		dev_weights[transposed_ind] += learning_rate*temp_value/(1e-5 + sqrt(gradient_sum[transposed_ind]));
		//NOTE : MAX SCAMM
		// dev_weights[transposed_ind] += learning_rate*temp_value/(1 + sqrt(gradient_sum[transposed_ind]));
	}
}
