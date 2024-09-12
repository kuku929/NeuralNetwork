cmake_minimum_required(VERSION 3.22)
project(rl LANGUAGES CXX CUDA)

#setting flags
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# change to your architecture for best performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -g -G")
find_package(CUDAToolkit REQUIRED)

#include headers
include_directories(include/)
include_directories(utils/)

#adding files
add_library(optimizer ./src/optimizer.cu)
add_library(datareader ./utils/data_reader.cpp)
add_library(neuralnet ./src/cuNeuralNetwork.cu)
add_library(activation ./src/activation.cu)
add_library(layer ./src/layer.cu)
add_library(kernel ./src/kernel.cu)
add_library(loss ./src/loss.cu)
add_library(softmax ./src/softmax.cu)
add_library(debug_util ./utils/debug.cu)

set_source_files_properties(mnist.cpp PROPERTIES LANGUAGE CUDA)
add_executable(main mnist.cpp)
target_link_libraries(layer kernel debug_util)
target_link_libraries(optimizer layer)
target_link_libraries(neuralnet activation optimizer loss softmax)
target_link_libraries(main neuralnet datareader)

#cuda files settings
set_target_properties(neuralnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# debug mode
add_compile_definitions("$<$<CONFIG:DEBUG>:-DDEBUG>")
