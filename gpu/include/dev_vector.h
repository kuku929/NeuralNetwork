/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : definition of the dev_vector class
 * 				a general utility class with
 * 				device memory management
 */

#pragma once
#include "basic_matrix.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

inline bool check_success(cudaError_t result, int line, std::string filename = __FILE_NAME__)
{
    /*
     * @brief
     * pretty print if copy/malloc to device was successful
     *when in Debug mode, the program also returns the line
     * which called the malloc/copy.
     */
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
        /*
         * @brief copy memory from host to device
         * @parms host pointer to copy from, amount to copy
         */
        size_t min = std::min(size, this->size());

        cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
#if defined(DEBUG)
        if (!check_success(result, line, file))
        {
            throw std::runtime_error("failed to copy to device!");
        }
#endif
    }

    void get(T *dest, size_t size, int line = __LINE__)
    {
        /*
         * @brief copy memory from device to host
         * @params host pointer to copy to, amount to copy
         */
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
        // for self-assignment
        if (start_ == second.start_)
            return this;
        free();
        allocate(second.size(), __LINE__, __FILE_NAME__);
        set(second.start_, second.size(), __LINE__, __FILE_NAME__);
        return *this;
    }

  private:
    void allocate(size_t size, int line = __LINE__, std::string file = __FILE_NAME__)
    {
        /*
         * @brief calls cudaMalloc with appropriate error checking if in Debug mode
         * 		  called when object is instantiated
         * @params size to malloc
         */
        cudaError_t result = cudaMalloc((void **)&start_, size * sizeof(T));
#if defined(DEBUG)
        if (!check_success(result, line, file))
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to allocate memory on device!");
        }
#endif
        end_ = start_ + size;
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
