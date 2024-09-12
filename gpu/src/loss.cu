#include "basic_matrix.h"
#include "dev_vector.h"
#include "loss.h"
using namespace nnet;

#if defined(DEBUG)
#define shared_dev_vector(a, b) shared_dev_vector(a, b, __LINE__, __FILE_NAME__)
#define set(a, b) set(a, b, __LINE__, __FILE_NAME__)
#endif

template <typename loss_func>
__global__ void find_loss_T(float *prediction, float *actual, float *output, size_t no_of_samples,
                            size_t size, loss_func loss)
{
    int col = threadIdx.x;
    float temp;
    for (int i = col; i <= (size - 1) * no_of_samples + col; i += no_of_samples)
    {
        temp += loss(prediction[i], actual[i]);
    }
    output[col] = temp;
}

template <typename loss_func>
__global__ void find_loss_(float *prediction, float *actual, float *output, size_t no_of_samples,
                           size_t size, loss_func loss)
{
    /*
     * am i tripping or this function is wrong??
     * true output has (size, no_of_samples) dim
     * right??
     */
    int row = threadIdx.x;
    float temp;
    for (int i = row * size; i < (row + 1) * size; ++i)
    {
        temp += loss(prediction[i], actual[i]);
    }
    output[row] = temp;
}

template <typename loss_func>
__global__ void find_loss_derivative_(float *prediction, float *actual, float *output,
                                      size_t no_of_samples, size_t size, loss_func loss_deriv)
{
    // NOTE : i am transposing right now, but need to think of a better way. row
    // is the row in input BROOOOOOOOOOOOOO THIS IS SHIT I NEED TO DO SOMETHING!
    // figuring out the correct dimensions is a PAIN IN THE ASS
    // think of a way so that my code is dimension independant or they are
    // implicitly handled OVER AND OUT -- graph theory tmrw ;<
    // Damn, I had a panic attack right here,. futures bad-08/09/24

    int row = threadIdx.x;
    int col = blockIdx.x;
    int index_in_matrix = row * no_of_samples + col;
    // transposing
    output[col * size + row] = loss_deriv(prediction[index_in_matrix], actual[index_in_matrix]);
}

dev_ptr MSELoss::find_loss(dev_vector<float> &prediction, dev_vector<float> &actual,
                           size_t no_of_samples)
{
    // output should be of size no_of_samples,1
    dev_ptr output = shared_dev_vector(no_of_samples, 1);
    dim3 dim_block(no_of_samples, 1);
    find_loss_T<<<1, dim_block>>>(prediction.data(), actual.data(), output->data(), no_of_samples,
                                  size, mse_loss_func_);
    cudaDeviceSynchronize();
    return output;
}

dev_ptr MSELoss::loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual,
                                 size_t no_of_samples)
{
    auto output = shared_dev_vector(no_of_samples, size);
    dim3 dim_block(size, 1);
    dim3 dim_grid(no_of_samples, 1);
    find_loss_derivative_<<<dim_grid, dim_block>>>(prediction.data(), actual.data(), output->data(),
                                                   no_of_samples, size, mse_loss_der_);
    cudaDeviceSynchronize();
    return output;
}

dev_ptr CrossEntropyLoss::find_loss(dev_vector<float> &prediction, dev_vector<float> &actual,
                                    size_t no_of_samples)
{
    dev_ptr output = shared_dev_vector(no_of_samples, 1);
    dim3 dim_block(size, 1);
    dim3 dim_grid(no_of_samples, 1);
    find_loss_T<<<1, dim_block>>>(prediction.data(), actual.data(), output->data(), no_of_samples,
                                  size, cross_entropy_loss_func_);
    cudaDeviceSynchronize();
    return output;
}

dev_ptr CrossEntropyLoss::loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual,
                                          size_t no_of_samples)
{
    auto output = shared_dev_vector(no_of_samples, size);
    dim3 dim_block(size, 1);
    dim3 dim_grid(no_of_samples, 1);
    find_loss_derivative_<<<dim_grid, dim_block>>>(prediction.data(), actual.data(), output->data(),
                                                   no_of_samples, size, cross_entropy_loss_der_);
    cudaDeviceSynchronize();
    return output;

    ////debug
    // basic_matrix<float> o(size, no_of_samples);
    // cudaMemcpy(o.data(), prediction_softmax.begin(), sizeof(float) * o.size,
    //            cudaMemcpyDeviceToHost);
    // o.show();
}
