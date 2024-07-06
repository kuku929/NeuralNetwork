cmake_minimum_required(VERSION 3.22)
project(rl LANGUAGES CXX CUDA)

#setting flags
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")
find_package(CUDAToolkit REQUIRED)

#include headers
include_directories(include/)

#adding files
add_library(neuralnet ./src/cuNeuralNetwork.cu)
add_library(layer ./src/layer.cu)
add_library(kernel ./src/kernel.cu)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(main main.cpp)
target_link_libraries(layer kernel)
target_link_libraries(neuralnet layer)
target_link_libraries(main neuralnet)

#cuda files settings
set_target_properties(neuralnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET kernel PROPERTY CUDA_ARCHITECTURES native)

