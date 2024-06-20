#include <cstdio>
#include <vector>
#include <string>
// model parameters
extern float INITIAL_LEARNING_RATE;
extern float GRAD_DECAY;
extern int OPTIMIZER;

//Network parameters
extern std::vector<size_t> LAYER_SIZES;
extern std::vector<std::string> ACTIVATION_FUNCTIONS;
