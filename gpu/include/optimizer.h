#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "dev_vector.h"
#include "layer.h"
#include <memory>

namespace nnet
{
class Layer;
class Optimizer
{
  public:
    Optimizer() = default;
    Optimizer(float learning_rate) : learning_rate(learning_rate)
    {
    }
    virtual ~Optimizer()
    {
    }
    virtual void update_weights(Layer *layer, const dev_vector<float> &layer_delta,
                                std::shared_ptr<dev_vector<float>> layer_output,
                                const size_t no_of_samples) = 0;
    virtual void update_bias(Layer *layer, const dev_vector<float> &layer_delta,
                             const size_t no_of_samples) = 0;
    // virtual void update(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>>
    // layer_output, const size_t no_of_samples);
  protected:
    virtual std::shared_ptr<Optimizer> clone() = 0;
    virtual void initialize(const Shape &shape) = 0;
    // std::shared_ptr<nnet::Layer> layer;
    float learning_rate;
    friend class Layer;
};

class RMSProp : public Optimizer
{
  public:
    RMSProp(float learning_rate, float beta);
    RMSProp(RMSProp &optimizer) : beta(optimizer.beta), Optimizer(optimizer.learning_rate)
    {
    }
    ~RMSProp() = default;

    void update_weights(Layer *layer, const dev_vector<float> &layer_delta,
                        std::shared_ptr<dev_vector<float>> layer_output,
                        const size_t no_of_samples) override;
    void update_bias(Layer *layer, const dev_vector<float> &layer_delta,
                     const size_t no_of_samples) override;

  protected:
    std::shared_ptr<Optimizer> clone() override;
    void initialize(const Shape &shape) override;
    // void update(const dev_vector<float> &layer_delta, std::shared_ptr<dev_vector<float>>
    // layer_output, const size_t no_of_samples) override;
  private:
    float beta;
    std::shared_ptr<dev_vector<float>> dev_grad_weights_;
    std::shared_ptr<dev_vector<float>> dev_grad_bias_;
};
} // namespace nnet
#endif // OPTIMIZER_H
