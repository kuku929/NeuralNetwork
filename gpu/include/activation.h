#pragma once
#include <dev_vector.h>
namespace activation{
class ActivationLayer{ //base class, forward_activate and back_activate need to be overridden for derived classes
	public:
		virtual void forward_activate(dev_vector<float> &input, dev_vector<float> &output)=0; //make it pure virtual, thus *requiring* override
		virtual void back_activate(dev_vector<float> &input, dev_vector<float> &output)=0;
		ActivationLayer() = default;
		virtual ~ActivationLayer(){};
};

class Linear : public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output) override{
			//todo : make this faster, redundant copy
			if(output.data() == input.data())
				return;
			output = input;
		}

		void back_activate(dev_vector<float> &input, dev_vector<float> &output) override{
			if(output.data() == input.data())
				return;
			output = input;
		}
		Linear() = default;
		~Linear() = default; 
};

class Sigmoid : public ActivationLayer{
	public:
		void forward_activate(dev_vector<float> &input, dev_vector<float> &output) override{
			//todo : finish this
			dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dim_grid((nrows + dim_block.x - 1)/dim_block.x, (ncols + dim_block.y - 1)/dim_block.y);
		}

		void back_activate(dev_vector<float> &input, dev_vector<float> &output) override{
		}
		Sigmoid() = default;
		~Sigmoid() = default; 
	private:
		__device__ sigmoid_(const float &input, float &output){
			float e_x = exp(input); 
			output = e_x/(1+e_x);
		}
};
}
