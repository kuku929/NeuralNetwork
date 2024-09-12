#pragma once
#include "basic_matrix.h"
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

inline bool check_success(cudaError_t result, int line, std::string filename = __FILE_NAME__)
{
    if (result != cudaSuccess)
    {
        std::cerr << "in " << filename << " at " << line << " : " << cudaGetErrorString(result)
                  << std::endl;
        return false;
    }
    return true;
}
template <class T> class dev_vector
{
  public:
    // constructors
    explicit dev_vector() : start_(0), end_(0)
    {
    }
    explicit dev_vector(size_t size, int line = __LINE__, std::string file = __FILE_NAME__)
    {
        allocate(size, line, file);
    }

    explicit dev_vector(const std::vector<T> &host_vector, int line = __LINE__,
                        std::string file = __FILE_NAME__)
    {
        allocate(host_vector.size(), line, file);
        set(host_vector.data(), host_vector.size(), line, file);
    }

    explicit dev_vector(const basic_matrix<T> &host_matrix, int line = __LINE__,
                        std::string file = __FILE_NAME__)
    {
        /*
         *copy the whole matrix
         */
        allocate(host_matrix.size, line, file);
        set(host_matrix.data(), host_matrix.size, line, file);
    }

    ~dev_vector()
    {
        free();
    }

    const T *begin() const
    {
        return start_;
    }

    __host__ __device__ T *data() const
    {
        return start_;
    }

    T *end() const
    {
        return end_;
    }

    size_t size() const
    {
        return end_ - start_;
    }

    void set(const T *src, size_t size, int line = __LINE__, std::string file = __FILE_NAME__)
    {
        size_t min = std::min(size, this->size());

        cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
#if defined(DEBUG)
        if (!check_success(result, line, file))
        {
            throw std::runtime_error("failed to copy to host!");
        }
#endif
    }

    void get(T *dest, size_t size, int line = __LINE__)
    {
        size_t min = std::min(size, this->size());
        cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
#if defined(DEBUG)
        if (!check_success(result, line))
        {
            throw std::runtime_error("failed to copy to host!");
        }
#endif
    }

    __host__ dev_vector<float> &operator=(const dev_vector<T> &second)
    {
        // this messes with dev_weights in layer class for some reason
        if (second.size() > size())
        {
            // throw std::runtime_error("dev vector sizes are unequal!");
            start_ = 0;
            end_ = 0;
            allocate(second.size(), __LINE__);
        }

        cudaError_t result =
            cudaMemcpy(start_, second.data(), sizeof(T) * second.size(), cudaMemcpyDeviceToDevice);
#if defined(DEBUG)
        if (!check_success(result, __LINE__))
        {
            throw std::runtime_error("failed to copy to host!");
        }
#endif
        return *this;
    }

  private:
    void allocate(size_t size, int line = __LINE__, std::string file = __FILE_NAME__)
    {
        cudaError_t result = cudaMalloc((void **)&start_, size * sizeof(T));
#if defined(DEBUG)
        if (!check_success(result, line, file))
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to copy to host!");
        }
#endif
        end_ = start_ + size;
    }

    void extend(size_t size, int line = __LINE__)
    {
        cudaError_t result = cudaMalloc((void **)&start_, size * sizeof(T));
#if defined(DEBUG)
        if (!check_success(result, line))
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to copy to host!");
        }
#endif
        end_ = start_;
    }

    void push(const T *src, size_t size, int line = __LINE__)
    {
        cudaError_t result = cudaMemcpy(end_, src, size * sizeof(T), cudaMemcpyHostToDevice);
#if defined(DEBUG)
        if (!check_success(result, line))
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to copy to host!");
        }
#endif
        end_ += size;
    }

    void free()
    {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T *start_;
    T *end_;
};
