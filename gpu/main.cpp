#include "activation.h"
#include "basic_matrix.h"
#include "cuNeuralNetwork.h"
#include "optimizer.h"
#include <chrono>
#include <iostream>
#include <vector>
using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1>>;
int main()
{
    // testing xor
    float learning_rate = 0.01f;
    float beta = 0.9f;
    uint epochs = 1000;
    nnet::Network net;
    // layer 1
    net.add_layer(2, 4, nnet::RMSProp(learning_rate, beta));
    // activation 1
    net.add_activation(nnet::ReLU(4));
    ////optimizer 1
    // auto optim = nnet::RMSProp(net.layers.back(), learning_rate, beta);
    // net.add_optimizer(optim);

    // layer 2
    net.add_layer(4, 2, nnet::RMSProp(learning_rate, beta));
    // activation 2
    // net.add_activation(nnet::Linear(2));
    net.add_softmax(2);
    ////optimizer 2
    // auto optim2 = nnet::RMSProp(net.layers.back(), learning_rate, beta);
    // net.add_optimizer(optim2);

    auto loss = nnet::CrossEntropyLoss(2);
    net.add_loss(loss);

    // net.print_weights();

    basic_matrix<float> test(2, 4);
    test.get(0, 0) = 1;
    test.get(1, 0) = 0;

    test.get(0, 1) = 1;
    test.get(1, 1) = 1;

    test.get(0, 2) = 0;
    test.get(1, 2) = 1;

    test.get(0, 3) = 0;
    test.get(1, 3) = 0;

    // basic_matrix<float> o = net.forward_pass(test);
    // o.show();

    // std::cout << "----------------------------\n";

    basic_matrix<float> op(2, 4);
    op.get(0, 0) = 1;
    op.get(1, 0) = 0;

    op.get(0, 1) = 0;
    op.get(1, 1) = 1;

    op.get(0, 2) = 1;
    op.get(1, 2) = 0;

    op.get(0, 3) = 0;
    op.get(1, 3) = 1;

    // basic_matrix<float> sub_test(2,1);
    // basic_matrix<float> sub_op(1,1);

    srand(1);
    for (int i = 0; i < epochs; ++i)
    {
        // int ind = rand() % 4;
        // sub_test=test.get_col(ind);
        // sub_op=op.get_col(ind);

        // std::cout << "INPUT : \n";
        // sub_test.show();

        // std::cout << "PREDICATION Before:\n";
        // o = net.forward_pass(test);
        // o.show();
        std::cout << i << " ";
        net.backward_pass(test, op);

        // std::cout << "PREDICATION After:\n";
        // o = net.forward_pass(test);
        // o.show();
    }

    std::cout << "final output :\n";
    auto o = net.forward_pass(test);
    o.show();
}
