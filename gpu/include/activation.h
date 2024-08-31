#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "dev_vector.h"

namespace nnet{
class ActivationLayer{ //base class, forward_activate and back_activate need to be overridden for derived classes
	public:
		virtual void forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples)=0; //make it pure virtual, thus *requiring* override
		virtual void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples)=0;
		ActivationLayer() = default;
		ActivationLayer(int s): size(s){};
		virtual ~ActivationLayer(){};
	protected:
		int size;
};

class Linear : public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples) override;
		void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples) override;
		Linear() = default;
		Linear(int s): ActivationLayer(s){};
		~Linear() = default; 
};

class Sigmoid : public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples) override;
		void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples) override;
		Sigmoid() = default;
		Sigmoid(int s): ActivationLayer(s){};
		~Sigmoid() = default; 
		struct SigmoidForward{
			__host__ __device__ float operator()(float input){
				float e_x = exp(input); 
				return e_x/(1+e_x);
			}
		}f_sigmoid_;
	
		struct SigmoidBack{
			__host__ __device__ float operator()(float input, float layer_output){
				return layer_output*(1-layer_output)*input;
			}
		}b_sigmoid_;
};

class ReLU: public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output, int no_of_samples) override;
		void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output, int no_of_samples) override;
		ReLU() = default;
		ReLU(int s): ActivationLayer(s){};
		~ReLU() = default; 
		struct ReLUForward{
			__host__ __device__ float operator()(float input){
				if(input > 0)
					return input;
				return 0;
			}
		}f_relu_;
	
		struct ReLUBack{
			__host__ __device__ float operator()(float input, float layer_output){
				if(layer_output > 0)
					return input;
				return 0;
			}
		}b_relu_;
};
}

#endif
