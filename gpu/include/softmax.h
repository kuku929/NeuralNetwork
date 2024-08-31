#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "dimension.h"
#include "dev_vector.h"
namespace nnet{
class Softmax{
public:
	Softmax(size_t size): dim(size){}
	~Softmax() = default;
	void forward_pass(const dev_vector<float> &input, dev_vector<float> &output, size_t no_of_samples);
	void back_pass(const dev_vector<float> &input, dev_vector<float> &output, const size_t no_of_samples);
private:
	Dimension dim;
};
}

#endif //SOFTMAX_H
