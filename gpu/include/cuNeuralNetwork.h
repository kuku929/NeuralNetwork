#ifndef NEURALNET_H
#define NEURALNET_H
#include "activation.h"
#include "base_layer.h"
#include "basic_matrix.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "softmax.h"
#include <cstdio>
#include <iostream>
#include <vector>

namespace nnet
{
class Network
{
  public:
    // right now you need an optimizer for each layer, maybe there is a better way?
    // in the future, the back_pass of the network will be written by user in main
    // std::vector<nnet::Layer> layers;
    // todo : convert all these to m_*
    // NOTE : softmax should be handled better

    Network()
    {
    }
    void add_layer(size_t front_layer_size, size_t back_layer_size, Optimizer &&optimizer);
    /*
     * adds a layer to the network
     * input : size of the layer, activation function for that layer
     * returns : nothing
     */

    void add_activation(ActivationLayer &&activation_function);
    void add_optimizer(Optimizer &optimizer);
    void add_loss(Loss &loss);
    void add_softmax(size_t size);
    void print_weights(std::ostream &out = std::cout);
    /*
     * prints weights in an orderly manner
     * input : buffer to print to
     * returns : nothing
     */

    basic_matrix<float> forward_pass(basic_matrix<float> &&input);
    basic_matrix<float> forward_pass(basic_matrix<float> &input);
    /*
     * one forward pass through the layers
     * input : input data to the neural network
     * returns : output of the network as a vector
     */
    void backward_pass(basic_matrix<float> &input, basic_matrix<float> &true_output);
    /*
     * one backward pass through the layers
     * input : input data to the neural network, true output for the input given
     * returns : nothing
     */

  private:
    std::vector<std::shared_ptr<Layer>> layers; // optimizer will also own the memory
    std::vector<std::shared_ptr<dev_vector<float>>> layer_outputs;
    std::vector<std::shared_ptr<BaseLayer>> layer_stack;
    std::vector<nnet::ActivationLayer *> activation_layers;
    std::unique_ptr<Softmax> softmax;
    Loss *loss;
    size_t max_layer_size = 0;
};
} // namespace nnet
#endif // NEURALNET_H
