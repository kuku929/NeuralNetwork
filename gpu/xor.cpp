#include "cuNeuralNetwork.h"
#include <chrono>
#include <iostream>
using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1>>;
int main()
{
    float learning_rate = 0.01f;
    float beta = 0.9f;
    uint epochs = 1000;
    nnet::Network net;
    // layer 1
    net.add_layer(2, 4, nnet::RMSProp(learning_rate, beta));
    // activation 1
    net.add_activation(nnet::ReLU(4));

    // layer 2
    net.add_layer(4, 2, nnet::RMSProp(learning_rate, beta));
    net.add_softmax(2);

    auto loss = nnet::CrossEntropyLoss(2);
    net.add_loss(loss);

    basic_matrix<float> test(2, 4);
    test.get(0, 0) = 1;
    test.get(1, 0) = 0;

    test.get(0, 1) = 1;
    test.get(1, 1) = 1;

    test.get(0, 2) = 0;
    test.get(1, 2) = 1;

    test.get(0, 3) = 0;
    test.get(1, 3) = 0;

    basic_matrix<float> op(2, 4);
    op.get(0, 0) = 1;
    op.get(1, 0) = 0;

    op.get(0, 1) = 0;
    op.get(1, 1) = 1;

    op.get(0, 2) = 1;
    op.get(1, 2) = 0;

    op.get(0, 3) = 0;
    op.get(1, 3) = 1;

    for (int i = 0; i < epochs; ++i)
    {
        std::cout << i << " ";
        net.backward_pass(test, op);
    }

    std::cout << "final output :\n";
    auto o = net.forward_pass(test);
    o.show();
}
