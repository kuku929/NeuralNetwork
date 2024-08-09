#pragma once
#include "basic_matrix.h"
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <iostream>

template <class T>
class dev_vector{
	public:
		//constructors
		explicit dev_vector(): start_(0), end_(0){}
		explicit dev_vector(size_t size){
			allocate(size);
		}

		explicit dev_vector(const std::vector<T> &host_vector){
			allocate(host_vector.size());
			set(host_vector.data(), host_vector.size());
		}


		explicit dev_vector(const basic_matrix<T> &host_matrix){
			/*
			 *copy the whole matrix
			 */
			allocate(host_matrix.size);
			set(host_matrix.data(), host_matrix.size);
		}

		explicit dev_vector(const basic_matrix<T> &host_matrix, int row){
			/*
			 *copy one row from host to device
			 */
			allocate(host_matrix.nrows);
			set(*host_matrix.get(row, 0), host_matrix.nrows);
		}


		~dev_vector(){
			free();
		}

		const T* begin() const{
			return start_;
		}

		__host__ __device__ T* data() const{
			return start_;
		}


		T* end() const{
			return end_;
		}

		size_t size() const{
			return end_ - start_;
		}

		void set(const T* src, size_t size){
			size_t min = std::min(size, this->size());

			cudaError_t result = cudaMemcpy(start_, src, min*sizeof(T), cudaMemcpyHostToDevice);
			if(result != cudaSuccess){
				std::cerr << cudaGetErrorString(result);
				throw std::runtime_error("failed to copy to device!");
			}
		}



		void get(T* dest, size_t size){
			size_t min = std::min(size, this->size());
			cudaError_t result = cudaMemcpy(dest, start_, min*sizeof(T), cudaMemcpyDeviceToHost);
			if(result != cudaSuccess){
				std::cerr << cudaGetErrorString(result) << std::flush;
				throw std::runtime_error("failed to copy to host!");
			}

		}


		__host__ void operator=(const dev_vector<T> &second){
			//this messes with dev_weights in layer class for some reason
			if(second.size() > size()){
				//throw std::runtime_error("dev vector sizes are unequal!");
				start_=0;
				end_=0;
				allocate(second.size());
			}
			
			cudaError_t result =  cudaMemcpy(start_, second.data(), sizeof(T)*second.size(), cudaMemcpyDeviceToDevice);
			if(result != cudaSuccess){
				std::cerr << cudaGetErrorString(result);
				throw std::runtime_error("failed to copy vector!");
			}
		}

	private:
		void allocate(size_t size){
			cudaError_t result = cudaMalloc((void**)&start_, size*sizeof(T));
			if(result != cudaSuccess){
				start_=end_=0;
				std::cerr << cudaGetErrorString(result) << std::flush;
				throw std::runtime_error("failed to copy to host!");
			}
			end_=start_+size;
		}

		void extend(size_t size){
			cudaError_t result = cudaMalloc((void**)&start_, size*sizeof(T));
			if(result != cudaSuccess){
				start_=end_=0;
				std::cerr << cudaGetErrorString(result) << std::flush;
				throw std::runtime_error("failed to copy to host!");
			}
			end_=start_;
		}

		void push(const T* src, size_t size){
			cudaError_t result = cudaMemcpy(end_, src, size*sizeof(T), cudaMemcpyHostToDevice);
			if(result != cudaSuccess){
				std::cerr << cudaGetErrorString(result) << std::flush;
				throw std::runtime_error("failed to copy to device!");
			}
			end_+=size;
		}


		void free(){
			if(start_!=0){
				cudaFree(start_);
				start_=end_=0;
			}
		}

		T* start_;
		T* end_;
};
