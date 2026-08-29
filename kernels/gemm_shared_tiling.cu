#include <cuda_runtime.h>

#include "gemm.hpp"


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

    const int tx=threadIdx.x;
    const int ty=threadIdx.y;

    const int row=blockIdx.y*TILE_SIZE+ty;
    const int col=blockIdx,x*TILE_SIZE+tx;

    float sum=0.0f;
    //沿K维切分tile
    for(int tile_k=0;tile_k<K;tile_k+=TILE_SIZE){
        if(row<M&&tile_k+tx<K){
            tile_A[ty][tx]=A[row*K+tile_k+tx];
        }else{
            tile_A[ty][tx]=0.0f;
        }

        //loda data to shared memory
        if(col<N&&tile_k+ty<K){
            tile_B[ty][tx]=B[(tile_k + ty) * N +
                  col];
        }else{
            tile_B[ty][tx]=0.0f;
        }

        __syncthreads();

        //内积
        for(int k=0;k<TILE_SIZE;++k){
            sum +=
                tile_A[ty][k] *
                tile_B[k][tx];
        }

        __syncthreads();

        
    }
    if (row < M && col < N) {

        C[row * N + col] =
            alpha * sum +
            beta * C[row * N + col];
    }
}

void launcher_gemm_shared_tilling(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
){
    constexpr int TILE_SIZE=16;
    dim3 block(TILE_SIZE,TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);

    gemm_shared_tiling<TILE_SIZE><<<grid,block>>>(M,N,K,alpha,A,B,beta,C);

}
