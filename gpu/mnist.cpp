#include <iostream>
#include <fstream>
#include "basic_matrix.h"
#include "cuNeuralNetwork.h"
#include "activation.h"

int main(){

    //getting data
    std::ifstream train_images_in("../data/mnist_train_images_filtered");
    std::ifstream train_labels_in("../data/mnist_train_labels_filtered");

    int input_size = 784;
    int output_size = 10;
    int no_of_samples = 32;
    int epochs = 10;

    basic_matrix<float> train_images(input_size, no_of_samples);
    basic_matrix<float> train_labels(output_size, no_of_samples);

    for(int i=0; i < no_of_samples; ++i){
        for(int j=0; j < input_size; ++j){
            train_images_in >> train_images.get(j, i);
            train_labels_in >> train_labels.get(0, i);
        }
    }

    //neural network
    float learning_rate = 0.009;
    float beta = 0.9;
    nnet::Network net;
    //layer 1
    net.add_layer(input_size, 512);
    auto activ = nnet::Sigmoid(512);
    net.add_activation(activ);
    auto optim = nnet::RMSProp(net.layers.back(), learning_rate, beta);
    net.add_optimizer(optim);

    //layer 2
    net.add_layer(512, 512);
    auto activ1 = nnet::Sigmoid(512);
    net.add_activation(activ1);
    auto optim1 = nnet::RMSProp(net.layers.back(), learning_rate, beta);
    net.add_optimizer(optim1);

    //layer 3
    net.add_layer(512, output_size);
    auto activ2 = nnet::Sigmoid(output_size);
    net.add_activation(activ2);
    auto optim2 = nnet::RMSProp(net.layers.back(), learning_rate, beta);
    net.add_optimizer(optim2);

    std::cout << "before :\n";
    auto o = net.forward_pass(train_images.get_col(0));
    o.show();

    //training
    for(int i=0;i < epochs; ++i){
        net.backward_pass(train_images, train_labels);
        o = net.forward_pass(train_images.get_col(0));
        o.show();
    }

    //testing
    std::cout << "after :\n";
    o = net.forward_pass(train_images.get_col(0));
    o.show();
    std::cout << "actual values : \n";
    train_labels.get_col(0).show();

}