#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "base_layer.h"
#include "dev_vector.h"
#include <memory>

typedef std::shared_ptr<dev_vector<float>> dev_ptr;
namespace nnet
{
class ActivationLayer : public BaseLayer
{ // base class, forward_activate and back_activate need to be overridden for
    // derived classes
  public:
    // ActivationLayer(){}
    ActivationLayer(size_t size) : BaseLayer(size){};
    ActivationLayer(Shape &shape) : BaseLayer(shape){};
    virtual std::shared_ptr<ActivationLayer> clone() = 0;
    virtual ~ActivationLayer()
    {
    }
};

class Linear final : public ActivationLayer
{
  public:
    Linear(Linear &activation) : ActivationLayer(activation.dim)
    {
    }
    // Linear() = default;
    Linear(size_t s) : ActivationLayer(s){};
    ~Linear() = default;
    dev_ptr forward_pass(dev_ptr input, size_t no_of_samples) override;
    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;

    std::shared_ptr<ActivationLayer> clone() override
    {
        auto ptr = std::make_shared<Linear>(*this);
        return ptr;
    }
};

class Sigmoid final : public ActivationLayer
{
  public:
    // Sigmoid() = default;
    Sigmoid(Sigmoid &activation) : ActivationLayer(activation.dim)
    {
    }
    Sigmoid(size_t s) : ActivationLayer(s){};
    ~Sigmoid() = default;
    dev_ptr forward_pass(dev_ptr input, size_t no_of_samples) override;
    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;

    std::shared_ptr<ActivationLayer> clone() override
    {
        auto ptr = std::make_shared<Sigmoid>(*this);
        return ptr;
    }
    struct SigmoidForward
    {
        __host__ __device__ float operator()(float input)
        {
            float e_x = exp(-input);
            return 1.0 / (1 + e_x);
        }
    } f_sigmoid_;

    struct SigmoidBack
    {
        __host__ __device__ float operator()(float input, float layer_output)
        {
            return layer_output * (1 - layer_output) * input;
        }
    } b_sigmoid_;
};

class ReLU final : public ActivationLayer
{
  public:
    ReLU(ReLU &activation) : ActivationLayer(activation.dim)
    {
    }
    // ReLU() = default;
    ReLU(size_t s) : ActivationLayer(s){};
    ~ReLU() = default;
    dev_ptr forward_pass(dev_ptr input, size_t no_of_samples) override;
    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;

    std::shared_ptr<ActivationLayer> clone() override
    {
        auto ptr = std::make_shared<ReLU>(*this);
        return ptr;
    }
    struct ReLUForward
    {
        __host__ __device__ float operator()(float input)
        {
            if (input > 0)
                return input;
            return 0;
        }
    } f_relu_;

    struct ReLUBack
    {
        __host__ __device__ float operator()(float input, float layer_output)
        {
            if (layer_output > 0)
                return input;
            return 0;
        }
    } b_relu_;
};
} // namespace nnet

#endif
