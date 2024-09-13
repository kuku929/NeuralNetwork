/*
 *@Author: Krutarth Patel                                           
 *@Date: 13th september 2024
 *@Description : declaration of the Network class
 * 				 the end user imports this file in
 * 				 their script
 */

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
    Network()
    {
    }
    void add_layer(size_t front_layer_size, size_t back_layer_size, Optimizer &&optimizer);
    /*
     *@brief  adds a layer to the network stack
     *@params size of the layer, optimizer
     *@return void
     */
    void add_activation(ActivationLayer &&activation_function);
	/*
     *@brief  adds an activation layer to the network stack
     *@params activation function 
     *@return void
     */
    void add_loss(Loss &loss);
	/*
     *@brief  adds a loss function to the network stack
     *@params loss function
     *@return void
     */
    void add_softmax(size_t size);
	/*
     *@brief  adds a softmax layer to the network stack
     *@params size of the layer, usually the output size
     *@return void
     */
    void print_weights(std::ostream &out = std::cout);
    /*
     *@brief prints weights in an orderly manner
     *@params buffer to print to
     *@return void
     */
    basic_matrix<float> forward_pass(basic_matrix<float> &&input);
    /*
     * convenience function
     */
    basic_matrix<float> forward_pass(basic_matrix<float> &input);
    /*
     *@brief forward pass through the layers
     *@params input data with no_of_samples columns
     *@return output of the network
     */
    void backward_pass(basic_matrix<float> &input, basic_matrix<float> &true_output);
    /*
     *@brief backward pass through the layers
     *@params input data, true value
     *@returns : void, will update layer weights
     */

  private:
    std::vector<std::shared_ptr<Layer>> m_layers; 
    std::vector<std::shared_ptr<dev_vector<float>>> m_layer_outputs;
    std::vector<std::shared_ptr<BaseLayer>> m_layer_stack;
    std::unique_ptr<Softmax> m_softmax;
    Loss *loss;
};
} // namespace nnet
#endif // NEURALNET_H
