/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : declaration of the Optimizer class
 */

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
    /*
     * @brief base optimizer class
     * 		  derived classes should override
     * 		  update_bias and update_weights
     */
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

  protected:
    virtual std::shared_ptr<Optimizer> clone() = 0;
    /*
     * @brief
     * returns a copy of itself
     * done because virtual classes
     * cannot have virtual copy constructors
     */
    virtual void initialize(const Shape &shape) = 0;
    float learning_rate;
    friend class Layer;
};

class RMSProp : public Optimizer
{
    /*
     * @brief implementation of RMSProp
     */
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

  private:
    float beta;
    std::shared_ptr<dev_vector<float>> dev_grad_weights_;
    std::shared_ptr<dev_vector<float>> dev_grad_bias_;
};
} // namespace nnet
#endif // OPTIMIZER_H
