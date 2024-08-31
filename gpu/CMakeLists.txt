cmake_minimum_required(VERSION 3.22)
project(rl LANGUAGES CXX CUDA)

#setting flags
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
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

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(main main.cpp)
target_link_libraries(layer kernel)
target_link_libraries(optimizer kernel layer)
target_link_libraries(neuralnet layer activation optimizer loss)
target_link_libraries(main neuralnet activation optimizer datareader)

#cuda files settings
set_target_properties(neuralnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# should work but is slower
#set_property(TARGET kernel PROPERTY CUDA_ARCHITECTURES native)

