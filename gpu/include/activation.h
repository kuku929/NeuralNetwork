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
				output = input;
			}

			void back_activate(dev_vector<float> &input, dev_vector<float> &output) override{
				output = input;
			}
			Linear() = default;
			~Linear() = default; 
	};
}
