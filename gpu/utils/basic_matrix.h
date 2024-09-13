/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : definition of the basic_matrix class
 */

#pragma once
#include <array>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template <class T> class basic_matrix
{
    /*
     * simple matrix class
     */
  public:
    size_t nrows, ncols;
    size_t size;
    basic_matrix() : nrows(0), ncols(0), size(0) {};
    basic_matrix(int N, int M) : nrows(N), ncols(M), size(N * M)
    {
        data_.resize(size);
    }

    T *data()
    {
        /*
         * returns pointer to the first element of the vector
         * used while copying data from device memory to host
         */
        return &data_[0];
    }

    const T *data() const
    {
        /*
         * returns pointer to the first element of the vector
         * used when the matrix is not being updated
         */
        return data_.data();
    }

    T &get(int i, int j)
    {
        /*
         * returns pointer to the (i,j)th element of the matrix
         * used when the matrix is being updated
         */
        if (i >= nrows || j >= ncols)
        {
            throw std::out_of_range("index : " + std::to_string(i) + ' ' + std::to_string(j) +
                                    " out of bounds");
        }
        return data_[i * ncols + j];
    }
    const T &get(int i, int j) const
    {
        /*
         * returns pointer to the (i,j)th element of the matrix
         */
        if (i >= nrows || j >= ncols)
        {
            throw std::out_of_range("index : " + std::to_string(i) + ' ' + std::to_string(j) +
                                    " out of bounds");
        }
        return data_[i * ncols + j];
    }

    basic_matrix<T> get_col(int col)
    {
        /*
         * returns a col of type basic_matrix
         */
        basic_matrix<T> col_mat(nrows, 1);
        for (int i = 0; i < nrows; ++i)
        {
            col_mat.get(i, 0) = data_[col + i * ncols];
        }
        return col_mat;
    }

    void transpose()
    {
        std::vector<T> temp(data_);
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                // temp should be optimized away
                uint row = (j * nrows + i) / ncols;
                uint col = (j * nrows + i) % ncols;
                get(row, col) = temp[i * ncols + j];
            }
        }
        size_t rows = nrows;
        nrows = ncols;
        ncols = rows;
    }

    void min_max_normalize()
    {
        /*
         * min-max normalization
         */
        for (int i = 0; i < ncols; ++i)
        {
            T min_value = std::numeric_limits<T>::max(), max_value = std::numeric_limits<T>::min();
            T val;
            for (int j = 0; j < nrows; ++j)
            {
                val = get(j, i);
                if (val < min_value)
                    min_value = val;
                else if (val > max_value)
                    max_value = val;
            }
            for (int j = 0; j < nrows; ++j)
            {
                get(j, i) = (get(j, i) - min_value) / (max_value - min_value); //- 0.5;
            }
        }
    }

    basic_matrix<T> get_row(int row)
    {
        /*
         * returns a row of type basic_matrix
         */
        basic_matrix<T> row_mat(1, ncols);
        for (int i = 0; i < ncols; ++i)
        {
            row_mat.get(0, i) = data_[row * ncols + i];
        }
        return row_mat;
    }

    void add_row(std::vector<T> &row)
    {
        if (nrows != 0 && row.size() != ncols)
        {
            throw std::runtime_error("row dim does not match with matrix!");
        }
        if (nrows == 0)
            ncols = row.size();
        data_.insert(data_.end(), row.begin(), row.end());
        nrows++;
    }

    void show() const
    {
        /*
         * prints the matrix
         */
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                std::cout << data_[i * ncols + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

  protected:
    std::vector<T> data_;
};
