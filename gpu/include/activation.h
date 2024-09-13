/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : definition of the ActivationLayer class
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "base_layer.h"
#include "dev_vector.h"
#include <memory>

typedef std::shared_ptr<dev_vector<float>> dev_ptr;
namespace nnet
{
class ActivationLayer : public BaseLayer
{
    /*
     * @brief
     * special base class
     * the shape of the activation layer
     * is (n, n) like so:
     *
     * #--#
     * #--#
     * #--#
     *
     */
  public:
    ActivationLayer(size_t size) : BaseLayer(size) {};
    ActivationLayer(Shape &shape) : BaseLayer(shape) {};
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
    Linear(size_t s) : ActivationLayer(s) {};
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
    Sigmoid(Sigmoid &activation) : ActivationLayer(activation.dim)
    {
    }
    Sigmoid(size_t s) : ActivationLayer(s) {};
    ~Sigmoid() = default;
    dev_ptr forward_pass(dev_ptr input, size_t no_of_samples) override;
    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;

    std::shared_ptr<ActivationLayer> clone() override
    {
        auto ptr = std::make_shared<Sigmoid>(*this);
        return ptr;
    }
    // these structs are passed to the kernel.
    // they hold the expression for the functions
    // Forward is used during forward_pass
    // Backward is used during back_pass
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
    ReLU(size_t s) : ActivationLayer(s) {};
    ~ReLU() = default;
    dev_ptr forward_pass(dev_ptr input, size_t no_of_samples) override;
    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;

    std::shared_ptr<ActivationLayer> clone() override
    {
        auto ptr = std::make_shared<ReLU>(*this);
        return ptr;
    }
    // these structs are passed to the kernel.
    // they hold the expression for the functions
    // Forward is used during forward_pass
    // Backward is used during back_pass
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
