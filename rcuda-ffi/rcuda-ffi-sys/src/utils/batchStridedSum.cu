#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__
void batchStridedSum(
        float *input, float *output,
        int batchSize, int rows, int cols) {
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < (rows * cols); j++) {
            output[j] = output[j] + input[i * rows * cols + j];
        }
    }
}


extern "C" {
size_t cuBatchStridedSum(
        float *inputPtr, float *outputPtr,
        int batchSize, int rows, int cols
) {

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    batchStridedSum<<<1, 1>>>(inputPtr, outputPtr, batchSize, rows, cols);

    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    cudaDeviceSynchronize();
    return 0;

    Error:
    return 1;
}
} // extern C
