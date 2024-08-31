#include "data_reader.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace nnet;
DataReader::DataReader(std::string &&file_path):
    delimiter(',')
    {
        fin.open(file_path); 
        if (not fin) {
            throw std::ios_base::failure("file does not exist");
        }
        find_cols_();
        std::filebuf* pbuf = fin.rdbuf();
        std::size_t size = pbuf->pubseekoff(0, fin.end, fin.in);
        pbuf->pubseekpos(0,fin.in);
        data_.resize(size);
        // this will store with newline
        pbuf->sgetn(data_.data(), size);
        // std::cout << data_;
    }

template<typename T>
T DataReader::get_field(size_t row_ind, size_t col_ind)
{
    return tokenized_data_[row_ind].get_field<T>(col_ind);
}
void DataReader::read_row(size_t row_ind)
{
    tokenized_data_[row_ind].read_row();
}

void DataReader::tokenize_data()
{
    size_t l_pointer=0;
    DataRow row(ncols_);
    size_t field_ind = 0;
    // make sure the last field is also added
	std::cout << data_.back() << ' ' << std::flush;
    data_.push_back('\n');
    for(int r_pointer=0;r_pointer < data_.size(); ++r_pointer){
		// to avoid double \n's and eof problems
        if(data_[r_pointer] == '\n' && r_pointer!=l_pointer){
            data_[r_pointer] = '\0';
            row.add(field_ind, DataField(&data_[l_pointer], r_pointer-l_pointer));
            l_pointer = r_pointer+1;
            tokenized_data_.push_back(row);
            field_ind = 0;
        }
        if(data_[r_pointer] == delimiter){
            // null-terminated for easier reading
            data_[r_pointer] = '\0';
            row.add(field_ind, DataField(&data_[l_pointer], r_pointer-l_pointer));
            field_ind++;
            l_pointer = r_pointer+1;
        }
    }
}

void DataReader::find_cols_()
{
    ncols_=1;
    std::string temp;
    std::getline(fin, temp);
    for(auto ch : temp){
        if(ch == delimiter){ncols_++;}
    }    
    fin.seekg(0,std::ios_base::beg);
}

basic_matrix<float> DataReader::convert_to_matrix(size_t rows, size_t cols)
{
    basic_matrix<float> output(rows, cols);
    for(int i=0; i < rows; ++i){
        for(int j=0;j < cols; ++j){
            output.get(i, j) = get_field<float>(i, j);
        }
    }
	return output;
}
