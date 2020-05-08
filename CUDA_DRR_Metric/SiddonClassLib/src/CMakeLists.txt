project(SiddonGpu)

find_package(CUDA REQUIRED)

set(CMAKE_C_FLAGS "-fPIC")
set(CMAKE_CXX_FLAGS "-fPIC")

cuda_add_library(SiddonGpu SiddonLib/siddon_class.cu SiddonLib/siddon_class.cuh)
target_link_libraries(SiddonGpu cudart cudadevrt)
