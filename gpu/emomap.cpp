#include "cuNeuralNetwork.h"
#include "data_reader.h"
#include <iostream>

int main()
{
    int input_size = 2304;
    int output_size = 7;
    int no_of_samples = 64;
    int epochs = 300;

    // // getting data
    // std::ifstream train_images_in("../data/mnist_train_images_filtered");
    // std::ifstream train_labels_in("../data/mnist_train_labels_filtered");

    auto train_images_in = "../data/train_dataset.csv";

    // images
    nnet::DataReader reader(train_images_in, ' ');
    reader.tokenize_data();
    reader.drop_cols_until(1);
    auto imgs = reader.convert_to_matrix(no_of_samples, input_size);
    auto labels = reader.get_col<float>(reader.ncols() - 1, no_of_samples);
    basic_matrix<float> onehot_labels;
    for (auto label : labels)
    {
        std::vector<float> onehot_vec(output_size, 0.0f);
        onehot_vec[label] = 1;
        onehot_labels.add_row(onehot_vec);
    }
    onehot_labels.transpose();
    imgs.transpose();
    imgs.min_max_normalize();

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
    // auto o = net.forward_pass(imgs);
    // o.show();

    // training
    for (int i = 0; i < epochs; ++i)
    {
        net.backward_pass(imgs, onehot_labels);
        // o = net.forward_pass(train_images.get_col(0));
        // o.show();
    }

    //     // testing
    //     std::cout << "after :\n";
    //     o = net.forward_pass(imgs);
    //     float net_loss = 0.0f;
    //     for (int i = 0; i < o.nrows; ++i)
    //     {
    //         for (int j = 0; j < o.ncols; ++j)
    //         {
    //             net_loss = (o.get(i, j) - onehot_labels.get(i, j)) *
    //                        (o.get(i, j) - onehot_labels.get(i, j));
    //         }
    //     }
    //     std::cout << "MSE: " << net_loss << std::endl;
    //     // o.show();
    //     // std::cout << "actual values : \n";
    //     // train_labels.show();
}
