#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <string_view>

inline void cuda_check(cudaError_t error,std::string_view operation){
    if(error!=cudaSuccess){
        std::cerr<<"CUDA error "<<operation<<":"<<cudaGetErrorString(error)<<"\n";
        std::exit(EXIT_FAILURE);
    }
}