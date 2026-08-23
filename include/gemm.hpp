#pragma once

void launch_gemm_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                           float beta, float* C);