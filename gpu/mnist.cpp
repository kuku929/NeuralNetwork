#include "cuNeuralNetwork.h"
#include "data_reader.h"
#include <iostream>

int main()
{

    int input_size = 784;
    int output_size = 10;
    int no_of_samples = 64;
    int epochs = 1000;

    auto train_images_in = "../data/mnist_train_images_filtered";
    auto train_labels_in = "../data/mnist_train_labels_filtered";

    // images
    nnet::DataReader reader(train_images_in, ' ');
    reader.tokenize_data();
    auto train_images = reader.convert_to_matrix(no_of_samples, reader.ncols());
    train_images.transpose();

    // labels
    nnet::DataReader labels_reader(train_labels_in, ' ');
    labels_reader.tokenize_data();
    auto train_labels = labels_reader.convert_to_matrix(no_of_samples, labels_reader.ncols());
    train_labels.transpose();

    // neural network
    float learning_rate = 0.009f;
    float beta = 0.9f;
    nnet::Network net;
    size_t hidden_1 = 8;
    size_t hidden_2 = 8;

    // layer 1
    net.add_layer(input_size, hidden_1, nnet::RMSProp(learning_rate, beta));
    // activation 1
    net.add_activation(nnet::Sigmoid(hidden_1));

    // layer 2
    net.add_layer(hidden_1, hidden_2, nnet::RMSProp(learning_rate, beta));
    // activation 2
    net.add_activation(nnet::ReLU(hidden_2));

    // layer 3
    net.add_layer(hidden_2, output_size, nnet::RMSProp(learning_rate, beta));
    net.add_softmax(output_size);

    auto loss = nnet::CrossEntropyLoss(output_size);
    net.add_loss(loss);
    // std::cout << "before :\n";
    // auto o = net.forward_pass(train_images);
    // o.show();

    // training
    for (int i = 0; i < epochs; ++i)
    {
        net.backward_pass(train_images, train_labels);
    }

    // testing
    std::cout << "after :\n";
    auto o = net.forward_pass(train_images);
    float net_loss = 0.0f;
    for (int i = 0; i < o.nrows; ++i)
    {
        for (int j = 0; j < o.ncols; ++j)
        {
            net_loss =
                (o.get(i, j) - train_labels.get(i, j)) * (o.get(i, j) - train_labels.get(i, j));
        }
    }

    std::cout << "MSE: " << net_loss << std::endl;
    // o.show();
    // std::cout << "actual values : \n";
    // train_labels.show();
}
