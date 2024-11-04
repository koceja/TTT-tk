cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ttt_mlp_forward LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)


set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)



add_executable(ttt_mlp_forward ttt_mlp_forward.cu)
target_link_libraries(ttt_mlp_forward "${TORCH_LIBRARIES}")
set_property(TARGET ttt_mlp_forward PROPERTY CXX_STANDARD 20)
set_target_properties(ttt_mlp_forward PROPERTIES CUDA_ARCHITECTURES 90)


target_include_directories(ttt_mlp_forward PRIVATE /code/include)
