// #include "activation.h"
// #include "basic_matrix.h"
#include "cuNeuralNetwork.h"
#include "data_reader.h"
// #include "layer.h"
// #include "optimizer.h"
#include <iomanip>
#include <iostream>
#include <vector>
int main()
{
    // testing diabetes

    // getting data
    nnet::DataReader reader("../data/diabetes.csv");
    reader.tokenize_data();
    auto data_mat = reader.convert_to_matrix(16, reader.ncols());
    auto input = reader.convert_to_matrix(16, reader.ncols() - 1);
    auto output = data_mat.get_col(data_mat.ncols - 1);
    input.min_max_normalize();
    input.transpose();
    output.transpose();
    std::vector<float> row(output.ncols);
    for (int i = 0; i < output.ncols; ++i)
    {
        row[i] = 1.0f - output.get(0, i);
        // std::cout << row[i] << ' ';
    }
    output.add_row(row);
    // input.show();
    // std::cout << "--------";
    // output.show();

    float learning_rate = 0.009f;
    float beta = 0.9f;
    uint epochs = 650;
    nnet::Network net;
    // layer 1
    net.add_layer(input.nrows, 8, nnet::RMSProp(learning_rate, beta));
    // activation 1
    net.add_activation(nnet::Sigmoid(8));

    // layer 2
    net.add_layer(8, 4, nnet::RMSProp(learning_rate, beta));
    // activation 2
    net.add_activation(nnet::ReLU(4));

    // layer 3
    net.add_layer(4, output.nrows, nnet::RMSProp(learning_rate, beta));
    net.add_softmax(output.nrows);

    auto loss = nnet::CrossEntropyLoss(output.nrows);
    net.add_loss(loss);
    for (int i = 0; i < epochs; ++i)
    {
        std::cout << i << ' ';
        if (i == epochs * 9 / 10)
            SWITCH_OPTIMIZER = 1;
        net.backward_pass(input, output);
    }
    net.print_weights();
    std::cout << "FINAL:" << std::endl;
    auto o = net.forward_pass(input);
    std::cout << std::setprecision(1);
    o.show();
    output.show();
}
