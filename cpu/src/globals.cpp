#include "globals.h"
#include <vector>
#include <string>
#include <cstdio>

float INITIAL_LEARNING_RATE=0.09f;
float GRAD_DECAY=0.9f;
int OPTIMIZER=0;

std::vector<size_t> LAYER_SIZES({2,4,1});
std::vector<std::string> ACTIVATION_FUNCTIONS({"ReLU","Linear"});
