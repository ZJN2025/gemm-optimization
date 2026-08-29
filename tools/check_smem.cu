// 探测本机 GPU 的共享内存上限（两种手段互相印证）：
// 1. 查询设备属性：cudaDevAttrMaxSharedMemoryPerBlock（静态上限）
//                    cudaDevAttrMaxSharedMemoryPerBlockOptin（opt-in 上限）
// 2. 实测：对带 extern __shared__ 的 kernel 用 cudaFuncSetAttribute 申请动态共享内存，
//    在 [48KB, opt-in 上限] 区间二分，找到实际能被接受的最大值并启动验证。
//
// 背景：CUTLASS 3-stage 配置（144KB smem）在本机启动失败（invalid argument），
// 用本程序定位到 RTX 5060 Laptop 的 opt-in 上限只有 101376 字节（99KB），
// 因此项目里 CUTLASS 参考基线改用了 2-stage（96KB）。

#include <cuda_runtime.h>

#include <cstdio>

__global__ void smem_probe_kernel(int* out) {
    extern __shared__ float smem[];

    smem[threadIdx.x] = threadIdx.x;
    out[threadIdx.x] = smem[threadIdx.x];
}

// 在 [min_bytes, max_bytes] 内二分找 cudaFuncSetAttribute 能接受的最大动态共享内存大小
int find_max_settable_smem(int min_bytes, int max_bytes) {
    int lo = min_bytes;
    int hi = max_bytes;

    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;

        const cudaError_t error = cudaFuncSetAttribute(
            smem_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mid);

        if (error == cudaSuccess) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    return lo;
}

int main() {
    int static_limit = 0;
    int optin_limit = 0;

    cudaDeviceGetAttribute(&static_limit, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    std::printf("static smem limit : %d bytes (%.1f KB)\n", static_limit,
                static_limit / 1024.0);
    std::printf("opt-in smem limit : %d bytes (%.1f KB)\n", optin_limit,
                optin_limit / 1024.0);

    // 实测：二分找出属性调用实际能接受的最大值（应等于 opt-in 上限）
    const int max_settable = find_max_settable_smem(static_limit, optin_limit);
    std::printf("measured max      : %d bytes (%.1f KB)\n", max_settable,
                max_settable / 1024.0);

    // 用测得的最大值真实启动一次验证
    int* d_out = nullptr;

    cudaMalloc(&d_out, 1024 * sizeof(int));

    cudaFuncSetAttribute(smem_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_settable);

    smem_probe_kernel<<<1, 1024, max_settable>>>(d_out);

    const cudaError_t launch_error = cudaGetLastError();
    cudaDeviceSynchronize();

    std::printf("launch at max     : %s\n", cudaGetErrorString(launch_error));

    cudaFree(d_out);

    return launch_error == cudaSuccess ? 0 : 1;
}
