#ifndef OPTIMIZER_H 
#define OPTIMIZER_H 
#include "dev_vector.h"
#include <memory>

namespace layer{
    class Layer;
}
namespace optimizer{
class Optimizer{
    public:
        Optimizer() = default;
        Optimizer(std::shared_ptr<layer::Layer> layer, float learning_rate): layer(layer), learning_rate(learning_rate){}
        virtual ~Optimizer(){};

        virtual void update_weights(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples)=0;
        virtual void update_bias(const dev_vector<float> &layer_delta, const size_t no_of_samples)=0;
        // virtual void update(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples);
    protected:
        std::shared_ptr<layer::Layer> layer;
        float learning_rate;
};

class RMSProp : public Optimizer{
    public:
        RMSProp() = default;
        RMSProp(std::shared_ptr<layer::Layer> layer, float learning_rate, float beta); 
        ~RMSProp() = default;

        void update_weights(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples) override;
        void update_bias(const dev_vector<float> &layer_delta, const size_t no_of_samples) override;
        // void update(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>> layer_output, const size_t no_of_samples) override;
    private:
        float beta;
        std::shared_ptr<dev_vector<float>> dev_grad_weights_;
        std::shared_ptr<dev_vector<float>> dev_grad_bias_;
};
}

#endif