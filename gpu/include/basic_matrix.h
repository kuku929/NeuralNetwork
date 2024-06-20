#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

template<class T>
class basic_matrix{
	/*
	 * simple matrix class
	 */
	public:
		size_t nrows, ncols;
		size_t size;
		basic_matrix(): nrows(0), ncols(0), size(0){};
		basic_matrix(int N, int M): nrows(N), ncols(M), size(N*M){
			data_.resize(size);
		}

		T* data(){
			/*
			 * returns pointer to the first element of the vector 
			 * used while copying data from device memory to host
			 */
			return &data_[0];
		}

		const T* data() const{
			/*
			 * returns pointer to the first element of the vector 
			 * used when the matrix is not being updated 
			 */
			return data_.data();
		}

		T& get(int i, int j){
			/*
			 * returns pointer to the (i,j)th element of the matrix 
			 * used when the matrix is being updated 
			 */
			return data_[i*ncols + j];
		}

		const T& get(int i, int j)const{
			/*
			 * returns pointer to the (i,j)th element of the matrix 
			 * used when the matrix is being updated 
			 */
			return data_[i*ncols + j];
		}

		void show() const{
			/*
			 * prints the matrix
			 */
			for(int i=0; i < nrows; ++i){
				for(int j=0; j < ncols; ++j){
					std::cout << data_[i*ncols + j] << ' ';
				}
				std::cout << '\n';
			}
		}

	protected:
		std::vector<T> data_;
};
