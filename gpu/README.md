# Demo
## XOR
using the following simple architecture
- OPTIMIZER: none
- ACTIVATION LAYER 1: ReLU
- ACTIVATION LAYER 2: Linear
- LOSS: MSE( mean squared error )
![nn(2)](https://github.com/user-attachments/assets/6111a01a-7b3f-41e5-9965-30d0b0a8c800)
We get the following loss vs epochs plot
![XOR](img/xor_nop_relu.png)

## MNIST
using the following architecture
- INPUT LAYER: 784
- ACTIVATION LAYER 1: Sigmoid
- HIDDEN LAYER 1: 8
- ACTIVATION LAYER 2: ReLU
- HIDDEN LAYER 2: 8
- ACTIVATION LAYER 3: Softmax
- LOSS: Cross entropy
- OPTIMIZER: RMSProp
![nn(3)](https://github.com/user-attachments/assets/624824ad-b578-4862-9673-0b499e05e37e)
We get the following loss vs epochs plot
![MNIST dataset](img/mnist_rmsprop_softmax.png)

# Usage
some examples have been provided. Implementation of the network
to train on XOR, PIMA Indian diabetes dataset and MNIST has been provided in the following files:
- xor.cpp : training on xor
- diab.cpp : training on PIMA Indian dataset
- mnist.cpp : training on MNIST dataset
