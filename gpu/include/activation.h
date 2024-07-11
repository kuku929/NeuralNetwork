#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "dev_vector.h"

namespace activation{
class ActivationLayer{ //base class, forward_activate and back_activate need to be overridden for derived classes
	public:
		virtual void forward_activate(dev_vector<float> &input, dev_vector<float> &output)=0; //make it pure virtual, thus *requiring* override
		virtual void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output)=0;
		ActivationLayer() = default;
		ActivationLayer(int s): size(s){};
		virtual ~ActivationLayer(){};
	protected:
		int size;
};

class Linear : public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output) override;
		void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output) override;
		Linear() = default;
		Linear(int s): ActivationLayer(s){};
		~Linear() = default; 
};

class Sigmoid : public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output) override;
		void back_activate(dev_vector<float> &input, dev_vector<float> &layer_output, dev_vector<float> &output) override;
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
			__host__ __device__ float operator()(float input, float output){
				return output*(1-output)*input;
			}
		}b_sigmoid_;
};
}

#endif
