#include "dev_vector.h"
#include "softmax.h"
using namespace nnet;

__global__ void softmax_(const float *input, float *output,
                         const size_t no_of_samples, const size_t size)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    double sum = 0;
    int index_in_matrix = row * no_of_samples + col;
    // note : each thread calculates this, make it shared?
    // iterate through the column
    for (int i = col; i < size * no_of_samples + col; i += no_of_samples)
    {
        // things can overflow, idk the fix
        sum += exp(input[i]);
    }
    output[index_in_matrix] = exp(input[index_in_matrix]) / sum;
}

dev_ptr Softmax::forward_pass(const dev_ptr input, const size_t no_of_samples)
{
    layer_input = input;
    layer_output =
        std::make_shared<dev_vector<float>>(get_shape().second * no_of_samples);
    dim3 dim_block(dim.second, 1);
    dim3 dim_grid(no_of_samples, 1);
    softmax_<<<dim_grid, dim_block>>>(input->data(), layer_output->data(),
                                      no_of_samples, dim.second);
    return layer_output;
}

dev_ptr Softmax::back_pass(const dev_ptr input, size_t no_of_samples)
{
    return input;
}
