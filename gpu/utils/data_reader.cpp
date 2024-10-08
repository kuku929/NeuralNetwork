/*
 *@Author: Krutarth Patel
 *@Date: 13th september 2024
 *@Description : definition of the DataReader class
 */

#include "data_reader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace nnet;
DataReader::DataReader(std::string &&file_path, char delim) : delimiter(delim), col_start_(0)
{
    fin.open(file_path);
    if (not fin)
    {
        throw std::ios_base::failure("file does not exist");
    }
    find_cols_();
    std::filebuf *pbuf = fin.rdbuf();
    std::size_t size = pbuf->pubseekoff(0, fin.end, fin.in);
    pbuf->pubseekpos(0, fin.in);
    data_.resize(size);
    // this will store with newline
    pbuf->sgetn(data_.data(), size);
}

DataReader::DataReader(std::string &data, char delim) : delimiter(delim), col_start_(0)
{
    data_ = data;
}

void DataReader::read_row(size_t row_ind)
{
    tokenized_data_[row_ind].read_row();
}

void DataReader::tokenize_data()
{
    size_t l_pointer = 0;
    DataRow row(ncols_);
    size_t field_ind = 0;
    // make sure the last field is also added
    data_.push_back('\n');
    for (int r_pointer = 0; r_pointer < data_.size(); ++r_pointer)
    {
        // to avoid double \n's and eof problems
        if (data_[r_pointer] == '\n')
        {
            data_[r_pointer] = '\0';
            if (r_pointer != l_pointer)
            {
                row.add(field_ind, DataField(&data_[l_pointer], r_pointer - l_pointer));
            }
            l_pointer = r_pointer + 1;
            tokenized_data_.push_back(row);
            field_ind = 0;
        }
        if (data_[r_pointer] == delimiter)
        {
            // null-terminated for easier reading
            data_[r_pointer] = '\0';
            if (r_pointer != l_pointer)
            {
                row.add(field_ind, DataField(&data_[l_pointer], r_pointer - l_pointer));
                field_ind++;
            }
            l_pointer = r_pointer + 1;
        }
    }
}

void DataReader::find_cols_()
{
    ncols_ = 1;
    std::string temp;
    std::getline(fin, temp);
    if (temp.back() == delimiter)
        ncols_ = 0;
    for (auto ch : temp)
    {
        if (ch == delimiter)
        {
            ncols_++;
        }
    }
    fin.seekg(0, std::ios_base::beg);
}

basic_matrix<float> DataReader::convert_to_matrix(size_t rows, size_t cols, sampler_ptr func)
{
    basic_matrix<float> output(rows, cols);
    size_t sample_ind;
    for (int i = 0; i < rows; ++i)
    {
        sample_ind = func(i);
        for (int j = 0; j < cols; ++j)
        {
            output.get(i, j) = get_field<float>(sample_ind, j);
        }
    }
    return output;
}
