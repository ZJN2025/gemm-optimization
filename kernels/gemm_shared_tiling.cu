#include <cuda_runtime.h>

#include "gemm.hpp"

constexpr int TILE_SIZE=16;
template <int TILE_SIZE>
__global__ void gemm_shared_tiling(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
){
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    const int row=blockIdx.y*TILE_SIZE+threadIdx.y;
    const int col=blockIdx,x*TILE_SIZE+threadIdx.x;

    float sum=0.0f;
    //沿K维切分tile
    for(int tile_k=0;tile_k<K;tile_k+=TILE_SIZE){
        //loda data to shared memory


        __syncthreads();

        //内积
        for(int k=0;k<TILE_SIZE;++k){


        }

        __syncthreads();

        




    }














}
