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
    int no_of_samples = 1024;

    basic_matrix<float> train_images(input_size, no_of_samples);
    basic_matrix<float> train_labels(output_size, no_of_samples);

    for(int i=0; i < no_of_samples; ++i){
        for(int j=0; j < input_size; ++j){
            train_images_in >> train_images.get(j, i);
            train_labels_in >> train_labels.get(0, i);
        }
    }

    //neural network
    net::Network net;
    net.add_layer(input_size, 512);
    auto activ = activation::ReLU(512);
    net.add_activation(activ);
    net.add_layer(512, 512);
    auto activ1 = activation::ReLU(512);
    net.add_activation(activ1);
    net.add_layer(512, output_size);
    auto activ2 = activation::ReLU(output_size);
    net.add_activation(activ2);

    //training
    net.backward_pass(train_images, train_labels);

    //testing
    auto o = net.forward_pass(train_images.get_col(0));
    o.show();
    std::cout << "actual values : \n";
    train_labels.get_col(0).show();

}