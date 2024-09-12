#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "base_layer.h"
#include "dev_vector.h"
#include <memory>

typedef std::shared_ptr<dev_vector<float>> dev_ptr;
namespace nnet
{
class Softmax : public BaseLayer
{
  public:
    // Softmax(): BaseLayer(){};
    Softmax(size_t size) : BaseLayer(size)
    {
    }
    ~Softmax() = default;
    dev_ptr forward_pass(const dev_ptr input,
                         const size_t no_of_samples) override;

    dev_ptr back_pass(const dev_ptr input, const size_t no_of_samples) override;
};
} // namespace nnet

#endif // SOFTMAX_H
