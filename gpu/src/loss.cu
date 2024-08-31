#include "basic_matrix.h"
#include "dev_vector.h"
#include "loss.h"
using namespace nnet;

template<typename loss_func>
__global__ void find_loss_(float *prediction, float *actual, float *output, size_t no_of_samples, size_t size, loss_func loss)
{
	int row = threadIdx.x;
	float temp;
	for(int i = row*size; i < (row+1)*size; ++i){
		temp += loss(prediction[i], actual[i]);
	}
	output[row] = temp;
}

template<typename loss_func>
__global__ void find_loss_derivative_(float *prediction, float *actual, float *output, size_t no_of_samples, size_t size, loss_func loss_derivative)
{
	// NOTE : i am transposing right now, but need to think of a better way. row is the row in input
	// BROOOOOOOOOOOOOO THIS IS SHIT I NEED TO DO SOMETHING!
	// figuring out the correct dimensions is a PAIN IN THE ASS
	// think of a way so that my code is dimension independant or they are implicitly handled
	// OVER AND OUT -- graph theory tmrw ;<
	int row = threadIdx.x;
	int col = blockIdx.x; 
	int index_in_matrix = row*no_of_samples + col;
	// transposing
	output[col*size + row] = loss_derivative(prediction[index_in_matrix], actual[index_in_matrix]);
}

void MSELoss::find_loss(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples)
{
	// output should be of size no_of_samples,1
	dim3 dim_block(no_of_samples, 1);
	find_loss_<<<1, dim_block>>>(prediction.data(), actual.data(), output.data(), no_of_samples, size, mse_loss_func_);
}

void MSELoss::loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) 
{
	dim3 dim_block(size, 1);
	dim3 dim_grid(no_of_samples, 1);
	find_loss_derivative_<<<dim_grid, dim_block>>>(prediction.data(), actual.data(), output.data(), no_of_samples, size, mse_loss_der_);
}

void CrossEntropyLoss::find_loss(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples)
{
	dev_vector<float> prediction_softmax(size*no_of_samples);
	dim3 dim_block(size, 1);
	dim3 dim_grid(no_of_samples, 1);
	find_loss_<<<1, dim_block>>>(prediction_softmax.data(), actual.data(), output.data(), no_of_samples, size, cross_entropy_loss_func_);
}

void CrossEntropyLoss::loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples)
{
	dev_vector<float> prediction_softmax(size*no_of_samples);
	dim3 dim_block(size, 1);
	dim3 dim_grid(no_of_samples, 1);
	find_loss_derivative_<<<dim_grid, dim_block>>>(prediction_softmax.data(), actual.data(), output.data(), no_of_samples, size, cross_entropy_loss_der_);

	basic_matrix<float> o(size, no_of_samples);
	cudaMemcpy(o.data(), prediction_softmax.begin(), sizeof(float)*o.size, cudaMemcpyDeviceToHost);
	o.show();

}
