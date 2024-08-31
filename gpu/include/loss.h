#include "dev_vector.h"
namespace nnet{
class Loss{
public:
	Loss(){};
	Loss(size_t size): size(size){};
	virtual void find_loss(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) = 0;
	virtual void loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) = 0;
protected:
	size_t size;
};

class MSELoss : public Loss{
public:
	MSELoss(){};
	MSELoss(size_t size) : Loss(size){};
	void find_loss(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) override;
	void loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) override;

	struct MSELoss_function{
		__host__ __device__ float operator()(float prediction, float actual){
			return (actual - prediction)*(actual - prediction)/2;
		}
	}mse_loss_func_;

	struct MSELoss_derivative{
		__host__ __device__ float operator()(float prediction, float actual){
			return (actual - prediction);
		}
	}mse_loss_der_;
};

class CrossEntropyLoss: public Loss{
public:
	CrossEntropyLoss(){};
	CrossEntropyLoss(size_t size) : Loss(size){};
	void find_loss(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) override;
	void loss_derivative(dev_vector<float> &prediction, dev_vector<float> &actual, dev_vector<float> &output, size_t no_of_samples) override;

	struct CrossEntropyLoss_function{
		__host__ __device__ float operator()(float prediction, float actual){
			// entropy
			// prediction should be softmaxxed
			if(actual == 1.0f) return -log(prediction);	
			return -log(1-prediction);
		}
	}cross_entropy_loss_func_;

	struct CrossEntropyLoss_derivative{
		__host__ __device__ float operator()(float prediction, float actual){
			// prediction should be softmaxxed
			// actual MUST be one hot encoded
			return actual - prediction;
		}
	}cross_entropy_loss_der_;
};
}


