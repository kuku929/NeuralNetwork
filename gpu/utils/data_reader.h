#include "basic_matrix.h"
#include <string>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <vector>

namespace nnet{
enum FieldTypes{
    INT,
    FLOAT,
    STRING
};

class DataField{
    public:
    DataField():start_(nullptr), size(0){}
    DataField(char *start, size_t size): start_(start), size(size){};
    template<typename T = std::string>
    T get(){
		// NOTE : no safety checks, use with caution
		T value;
		if constexpr(std::is_floating_point_v<T>){
			value = std::stof(start_);
		}
		else if constexpr(std::is_integral_v<T>){
			value = std::stoi(start_);
		}
		else{
			value = std::string(start_, size);
		}
		return value;
		
    }
    template<int type_enum>
    void read(){ 
        switch(type_enum)
        {
            case INT:
            {
                int data = std::stoi(start_);
                std::cout << data << ' ';
                break;
            }
            case FLOAT:
            {
                float data= std::stof(start_);
                std::cout << data << ' ';
                break;
            }
            case STRING:
            {
                for(int i=0;i < size; ++i){ std::cout << start_[i]; }
                break;
            }
        }
    }
    private:
    char *start_;
    size_t size;
    // FieldTypes type;
};

class DataRow{
    public:
    DataRow(){}
    DataRow(size_t size){ row_data_.resize(size); }
    size_t size(){ return row_data_.size(); }
    void push(DataField &field){ row_data_.push_back(field); }
    void add(size_t ind, DataField &&field)
    { 
        // out_of_range error thrown by at
        row_data_.at(ind) = field;
    }
    template<typename T>
    T get_field(size_t field_ind)
    {
        return row_data_.at(field_ind).get<T>();
    }
    void read_row()
    {
        for(auto &field : row_data_){
            field.read<FLOAT>(); 
        }
    }
    private:
    std::vector<DataField> row_data_;
};


class DataReader{
    public:
        DataReader(std::string &&file_path);
        void read_row(size_t row_ind);
        template<typename T>
        T get_field(size_t row_ind, size_t col_ind);
        void tokenize_data();
        basic_matrix<float> convert_to_matrix(size_t rows, size_t cols);
		size_t nrows(){ return tokenized_data_.size(); }
		size_t ncols(){ return ncols_; }

    private:
        void find_cols_();
        const size_t max_size = 10000;
        size_t ncols_;
        char delimiter;
        std::string data_;
        std::vector<DataRow> tokenized_data_;
        std::string file_path;
        std::ifstream fin; 
};
}
