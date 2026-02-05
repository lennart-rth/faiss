# experiments/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(faiss_experiments)

set(CMAKE_CXX_STANDARD 17)

# Locate the Faiss library in the parent directory's build folder
# We assume this file is in <root>/experiments/
set(FAISS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Include directories
include_directories(${FAISS_ROOT})

# Find OpenMP and BLAS (Required by Faiss)
find_package(OpenMP REQUIRED)
find_library(BLAS_LIB openblas) # On NixOS, typically libopenblas

# Define the executable
add_executable(efficiency efficiency.cpp)
add_executable(latency_recall latency_recall.cpp)

# Link libraries
# We link directly against the static libfaiss.a found in ../build/faiss/
target_link_libraries(efficiency 
    PRIVATE 
    "${FAISS_ROOT}/build/faiss/libfaiss.a" 
    ${BLAS_LIB}
    OpenMP::OpenMP_CXX
)

target_link_libraries(latency_recall 
    PRIVATE 
    "${FAISS_ROOT}/build/faiss/libfaiss.a" 
    ${BLAS_LIB}
    OpenMP::OpenMP_CXX
)