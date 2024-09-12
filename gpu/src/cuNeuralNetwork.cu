#include "activation.h"
#include "base_layer.h"
#include "basic_matrix.h"
#include "cuNeuralNetwork.h"
#include "dev_vector.h"
#include "kernel.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
// #define RMSPROP
using namespace nnet;

void Network::add_layer(size_t front_layer_size, size_t back_layer_size, Optimizer &&optimizer)
{
    /*
     * @arguments : size of the front set and back set, activation function for
     * the next layer
     *
     * @brief :
     * max_layer_size is required for later use in code while allocating
     * device memory.
     */
    // if layers have already been added
    // max layer size is zero when object initialized
    // static int calls = 0;
    if (front_layer_size == 0 || back_layer_size == 0)
    {
        throw std::invalid_argument("layer cannot have 0 neurons");
    }

    // if(!calls){
    //// the zeroeth layer will contain the input to the Network
    //// hack, but nothing wrong
    // std::shared_ptr<Layer> zero_layer_ptr =
    // std::make_shared<Layer>(front_layer_size);
    // layers.push_back(zero_layer_ptr);
    //}

    std::shared_ptr<Layer> layer_ptr =
        std::make_shared<Layer>(back_layer_size, front_layer_size, optimizer);
    layers.push_back(layer_ptr);
    layer_stack.push_back(layer_ptr);

    if (front_layer_size > max_layer_size)
        max_layer_size = front_layer_size;

    if (back_layer_size > max_layer_size)
        max_layer_size = back_layer_size;
    // calls++;
    return;
}

void Network::add_activation(ActivationLayer &&activation_function)
{
    auto layer_ptr = activation_function.clone();
    layer_stack.push_back(layer_ptr);
    activation_layers.push_back(&activation_function);
    return;
}

void Network::add_loss(Loss &loss)
{
    this->loss = &loss;
}

void Network::add_softmax(size_t size)
{
    layer_stack.push_back(std::make_shared<Softmax>(size));
}

basic_matrix<float> Network::forward_pass(basic_matrix<float> &&input)
{
    return forward_pass(input);
}

basic_matrix<float> Network::forward_pass(basic_matrix<float> &input)
{
    /*
     * @arguments : input matrix with dimensions : (input_layer_size,
     *no_of_samples) true output with dimensions : (output_layer_size,
     *no_of_samples)
     *
     * @brief:
     * iterates through layers calling the pass() function on each one
     * the dev_input and dev_output are swapped as the input and output
     * to the layers to avoid copying.
     * therefore, they are malloc'd with the maximum size at the beginning
     * Once the output is ready, it is copied to host
     */

    if (input.nrows != layer_stack.front()->get_shape().first)
    {
        throw std::invalid_argument("dimensions do not match!");
    }

    const size_t no_of_samples = input.ncols;

    basic_matrix<float> output(layer_stack.back()->get_shape().second, no_of_samples);

    //// copying to device
    // dev_vector<float> dev_input((max_layer_size)*no_of_samples);
    // dev_input.set(input.data(), input.size);
    // dev_vector<float> dev_output((max_layer_size)*no_of_samples);
    std::shared_ptr<dev_vector<float>> layer_input = std::make_shared<dev_vector<float>>(input);
    // forward pass
    // todo : use iterators
    for (int i = 0; i < layer_stack.size(); ++i)
    {

        //// debug
        // basic_matrix<float> test(layers[i]->ncols, no_of_samples);
        // cudaMemcpy(test.data(), dev_input.data(), sizeof(float)*test.size,
        // cudaMemcpyDeviceToHost); std::cout << "forward pass LAYER OUTPUT
        // :\n"; test.show();

        layer_input = layer_stack[i]->forward_pass(layer_input, no_of_samples);
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
     * @arguments : input matrix with dimensions : (input_layer_size,
     *no_of_samples) true output with dimensions : (output_layer_size,
     *no_of_samples)
     *
     * @brief:
     * finds the outputs of all the layers and stores them in layer_outputs
     * then, iterates backwards through layers calling the propagate() function
     *on each one the dev_input_delta and dev_output_delta are swapped as the
     *input and output to the layers to avoid copying. therefore, they are
     *malloc'd with the maximum size at the beginning
     */
    /*
     * the column in input will be for one training sample, we will do
     * input.ncols number of samples together
     */
    const size_t no_of_samples = input.ncols;
    // index i holds the layer outputs for layer i, each dev_vector will be a
    // matrix with rows = layers.nrows, cols = no of samples

    // //debug
    // std::vector<basic_matrix<float>> h_layer_outputs(layers.size()+1);
    // h_layer_outputs[0] = input;

    // //debug
    // for(int i=0; i < layer_outputs.size(); ++i){
    // 	cudaMemcpy(h_layer_outputs[i].data(), layer_outputs[i]->data(),
    // sizeof(float)*layer_outputs[i]->size(), cudaMemcpyDeviceToHost);
    // }
    // std::cout << "LAYER OUTPUTS\n";
    // for(auto const a : h_layer_outputs){
    // 	a.show();
    // 	std::cout << '\n';
    // }

    forward_pass(input);
    // finding intial delta
    dev_vector<float> dev_true_output(true_output);

    //// allocating maximum possible data to prevent needless copying
    // dev_vector<float> dev_input_delta(no_of_samples * max_layer_size);
    // dev_vector<float> dev_output_delta(no_of_samples * max_layer_size);

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((true_output.ncols + dim_block.x - 1) / dim_block.x,
                  (true_output.nrows + dim_block.y - 1) / dim_block.y);

    // // finding the difference and transposing the resulting matrix
    // // delta will have no_of_samples rows and layer size cols
    auto back_output =
        loss->loss_derivative(*(layer_stack.back()->get_output()), dev_true_output, no_of_samples);
    // no need to synchronize!

    // debug
    // std::cout << "loss: ";
    std::vector<float> o(no_of_samples);
    auto current_loss =
        loss->find_loss(*(layer_stack.back()->get_output()), dev_true_output, no_of_samples);
    cudaDeviceSynchronize();
    cudaMemcpy(o.data(), current_loss->begin(), sizeof(float) * o.size(), cudaMemcpyDeviceToHost);
    float delta_sum = 0;
    for (const auto out : o)
    {
        // std::cout << out << ' ';
        delta_sum += out;
    }
    // std::cout << '\n';
    // avg loss
    std::cout << delta_sum / no_of_samples << '\n' << std::flush;

    // propogating the delta and updating weights
    for (int i = layer_stack.size() - 1; i > -1; --i)
    {
        back_output = layer_stack[i]->back_pass(back_output, no_of_samples);

        // // debug
        // basic_matrix<float> output(no_of_samples, layers[i]->nrows);
        // auto result =
        //     cudaMemcpy(output.data(), back_output->begin(),
        //                sizeof(float) * output.size, cudaMemcpyDeviceToHost);
        // if (result != cudaSuccess)
        // {
        //     throw std::runtime_error("failed to copy to host!");
        // }
        // std::cout << "DELTA\n";
        // output.show();
        // std::cout << '\n';
    }

    // layer_stack[0]->back_pass(dev_output_delta, dev_input_delta,
    // no_of_samples); #if defined(RMSPROP)
    // optimizers[0]->update_weights(dev_output_delta, layer_outputs[i],
    // no_of_samples); optimizers[0]->update_bias(dev_output_delta,
    // no_of_samples); #else layers[0]->update(dev_output_delta, dev_input,
    // no_of_samples); #endif
}

void Network::print_weights(std::ostream &out)
{
    /*
     * prints the weights and biases
     * first the weights are printed
     * followed by the bias leaving a line in between
     */
    for (auto layer : layers)
    {
        layer->show();
        std::cout << '\n';
        layer->show_bias();
        std::cout << "*******\n";
    }
}
