#ifndef LAYER_H
#define LAYER_H

#include "base_layer.h"
#include "basic_matrix.h"
#include "debug.h"
#include "dev_vector.h"
#include <iostream>
#include <memory>
#include <vector>
typedef std::shared_ptr<dev_vector<float>> dev_ptr;
extern int SWITCH_OPTIMIZER;

namespace nnet
{
class Optimizer;
class RMSProp;

class Layer : public basic_matrix<float>, public BaseLayer
{
    /*
     * this class inherits from basic_matrix
     * contains weights and bias implemented as 1-D vectors
     */
  public:
    // Layer(): basic_matrix(), BaseLayer(){};
    Layer(int N, int M, Optimizer &optimizer);
    ~Layer() = default;
    dev_ptr forward_pass(const dev_ptr input, const size_t no_of_samples) override;
    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;

    void update(const dev_ptr &layer_delta, const size_t no_of_samples);

    void show_bias()
    {
        for (auto b : bias_)
        {
            std::cout << b << ' ';
        }
        std::cout << '\n';
    }

    float *bias()
    {
        /*
         * returns pointer to the first element of the vector
         * used while copying data from device memory to host
         */
        return &bias_[0];
    }

    const float *bias() const
    {
        /*
         * returns pointer to the first element of the vector
         * used when the matrix is not being updated
         */
        return bias_.data();
    }

    const float &get_bias(int row) const
    {
        /*
         * returns (i,j)th element in the matrix
         * used when value does not need to be changed
         */
        return bias_[row];
    }

    float &get_bias(int row)
    {
        /*
         * returns (i,j)th element in the matrix
         * used when value needs to be changed
         */
        return bias_[row];
    }
    const float &get_weights(int row, int col) const
    {
        /*
         * returns (i,j)th element in the matrix
         * used when value does not need to be changed
         */
        return data_[row * ncols + col];
    }

    float &get_weights(int row, int col)
    {
        /*
         * returns (i,j)th element in the matrix
         * used when value needs to be changed
         */
        return data_[row * ncols + col];
    }

    void copy_to_host();

    void show_dev()
    {
        std::cout << "show:\n";
        cudaMemcpy(this->data(), dev_weights_->data(), sizeof(float) * dev_weights_->size(),
                   cudaMemcpyDeviceToHost);
        this->show();
    }

  private:
    std::vector<float> bias_;
    dev_ptr dev_weights_;
    dev_ptr dev_bias_;
    std::shared_ptr<Optimizer> m_optimizer;
    friend class nnet::RMSProp;
};
} // namespace nnet
#endif // LAYER_H
