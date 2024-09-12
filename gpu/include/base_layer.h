#ifndef DIMENSION_H
#define DIMENSION_H

#include "dev_vector.h"
#include <iostream>
#include <memory>
namespace nnet
{
class Shape
{
  public:
    // Shape(): first(0), second(0){};
    Shape(size_t first, size_t second) : first(first), second(second) {};
    Shape(size_t first) : first(first), second(first) {};
    friend std::ostream &operator<<(std::ostream &os, const Shape &shape)
    {
        os << shape.first << ", " << shape.second << ' ';
        return os;
    }
    size_t first;
    size_t second;
};

class BaseLayer
{
  public:
    // BaseLayer(){};
    BaseLayer(size_t first, size_t second) : dim(first, second)
    {
    }
    BaseLayer(Shape &shape) : dim(shape)
    {
    }
    BaseLayer(size_t size) : dim(size) {};
    Shape get_shape()
    {
        return dim;
    }
    const std::shared_ptr<dev_vector<float>> get_output()
    {
        return layer_output;
    }
    virtual ~BaseLayer()
    {
    }
    virtual std::shared_ptr<dev_vector<float>> forward_pass(
        const std::shared_ptr<dev_vector<float>> input, const size_t no_of_samples) = 0;

    virtual std::shared_ptr<dev_vector<float>> back_pass(
        const std::shared_ptr<dev_vector<float>> input, const size_t no_of_samples) = 0;

    void make_shared_ptr_null()
    {
        layer_input = nullptr;
        layer_output = nullptr;
    }

  protected:
    Shape dim;
    std::shared_ptr<dev_vector<float>> layer_output = nullptr;
    std::shared_ptr<dev_vector<float>> layer_input = nullptr;
};
} // namespace nnet
#endif // DIMENSION_H
