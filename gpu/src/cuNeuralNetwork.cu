/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : definition of the Network class
 */

#include "activation.h"
#include "base_layer.h"
#include "basic_matrix.h"
#include "cuNeuralNetwork.h"
#include "dev_vector.h"
#include "kernel.h"
#include "loss.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
// #define RMSPROP
using namespace nnet;

void Network::add_layer(size_t front_layer_size, size_t back_layer_size, Optimizer &&optimizer)
{
    if (front_layer_size == 0 || back_layer_size == 0)
    {
        throw std::invalid_argument("layer cannot have 0 neurons");
    }

    std::shared_ptr<Layer> layer_ptr =
        std::make_shared<Layer>(back_layer_size, front_layer_size, optimizer);
    // storing linear layers separately for printing weights
    m_layers.push_back(layer_ptr);
    m_layer_stack.push_back(layer_ptr);
    return;
}

void Network::add_activation(ActivationLayer &&activation_function)
{
    auto layer_ptr = activation_function.clone();
    m_layer_stack.push_back(layer_ptr);
    return;
}

void Network::add_loss(Loss &loss)
{
    this->loss = &loss;
}

void Network::add_softmax(size_t size)
{
    m_layer_stack.push_back(std::make_shared<Softmax>(size));
}

basic_matrix<float> Network::forward_pass(basic_matrix<float> &&input)
{
    return forward_pass(input);
}

basic_matrix<float> Network::forward_pass(basic_matrix<float> &input)
{
    /*
     * @arguments : input matrix with dimensions : (input_layer_size, no_of_samples)
     * 				true output with dimensions : (output_layer_size, no_of_samples)
     *
     * @brief:
     * iterates through layers calling the forward_pass() function on each
     * layer
     */

    // input dimensions should match
    if (input.nrows != m_layer_stack.front()->get_shape().first)
    {
        throw std::invalid_argument("dimensions do not match!");
    }

    const size_t no_of_samples = input.ncols;

    basic_matrix<float> output(m_layer_stack.back()->get_shape().second, no_of_samples);
    std::shared_ptr<dev_vector<float>> layer_input = std::make_shared<dev_vector<float>>(input);

    // forward pass
    for (auto layer : m_layer_stack)
    {
        layer_input = layer->forward_pass(layer_input, no_of_samples);
    }

    // copying to host
    auto result = cudaMemcpy(output.data(), layer_input->begin(), sizeof(float) * output.size,
                             cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host!");
    }

    return output;
}

void Network::backward_pass(basic_matrix<float> &input, basic_matrix<float> &true_output)
{
    /*
     * @arguments : input matrix with dimensions : (input_layer_size, *no_of_samples)
     * 				true output with dimensions : (output_layer_size, *no_of_samples)
     *
     * @brief: iterates backwards through the stack
     * 			calling back_pass on each layer
     *
     */

    // the column in input will be for one training sample
    // doing input.ncols number of samples together
    const size_t no_of_samples = input.ncols;
    forward_pass(input);

    // finding intial delta
    dev_vector<float> dev_true_output(true_output);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((true_output.ncols + dim_block.x - 1) / dim_block.x,
                  (true_output.nrows + dim_block.y - 1) / dim_block.y);
    // finding the difference and transposing the resulting matrix
    auto back_output = loss->loss_derivative(*(m_layer_stack.back()->get_output()), dev_true_output,
                                             no_of_samples);

#if defined(DEBUG)
    std::vector<float> o(no_of_samples);
    MSELoss mse_loss(m_layer_stack.back()->get_shape().second);
    auto current_loss =
        mse_loss.find_loss(*(m_layer_stack.back()->get_output()), dev_true_output, no_of_samples);
    cudaDeviceSynchronize();
    cudaMemcpy(o.data(), current_loss->begin(), sizeof(float) * o.size(), cudaMemcpyDeviceToHost);
    float delta_sum = 0;
    for (const auto out : o)
    {
        delta_sum += out;
    }
    std::cout << delta_sum / no_of_samples << '\n' << std::flush;
#endif

    // propogating the delta and updating weights
    for (auto iter = m_layer_stack.rbegin(); iter != m_layer_stack.rend(); iter++)
    {
        back_output = (*iter)->back_pass(back_output, no_of_samples);
    }

    // cleaning up allocated memory
    for (auto layer : m_layer_stack)
    {
        // as all shared_ptrs are null'd,
        // memory will get freed
        layer->make_shared_ptr_null();
    }
}

void Network::print_weights(std::ostream &out)
{
    /*
     * prints the weights and biases
     * first the weights are printed
     * followed by the bias leaving a line in between
     */
    for (auto layer : m_layers)
    {
        layer->show();
        std::cout << '\n';
        layer->show_bias();
        std::cout << "*******\n";
    }
}
