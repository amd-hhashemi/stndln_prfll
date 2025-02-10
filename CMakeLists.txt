cmake_minimum_required(VERSION 3.10)
project(MyProject)

find_package(Torch REQUIRED PATHS "/sgl-workspace/libtorch")
#set(Torch_DIR /sgl-workspace/libtorch/share/cmake/Torch)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) 
#set(CMAKE_CUDA_FLAGS "-arch=sm_80 -rdc=true -O3 -std=c++17")

#add_compile_options(-O3 -std=c++17 -W -Wall -pedantic)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
add_library(my_cuda_lib SHARED src/cuda-wrapper.cu)
set_target_properties(my_cuda_lib PROPERTIES CUDA_ARCHITECTURES "80")
set_property(TARGET my_cuda_lib PROPERTY CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app my_cuda_lib ${TORCH_LIBRARIES})

# Optional: Set CUDA compile flags
set_target_properties(my_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(my_cuda_lib PROPERTIES CUDA_ARCHITECTURES "sm_90") # or specific architectures like "sm_70"


