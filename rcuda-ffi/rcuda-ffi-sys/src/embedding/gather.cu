#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__
void CuSubset(
        float *embedding, float *words, float *embeddingOutput,
        int phraseLength,
        int embedDimension,
        int vocabSize,
        int batchSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < (phraseLength * batchSize); i += stride) {
        // TODO: Add in boolean mask to remove branching
        for (int j = 0; j < embedDimension; j++) {
            int embedIndex = (j * vocabSize) + (int) words[i] - 1;
            int outputIndex = embedDimension * i + j;
            if (((int) words[i] != 0) && (embedIndex < vocabSize * embedDimension)) {
                embeddingOutput[outputIndex] = embedding[embedIndex];
            } else {
                embeddingOutput[outputIndex] = 0.0;
            }
        }
    }
}


extern "C" {
size_t cuGather(
        int embedDimension, int phraseLength,
        int vocabSize, int batchSize,
        float *src_ptr,
        float *weight_ptr,
        float *dest_ptr
) {

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    int blockSize = 256;
    int numBlocks = (phraseLength + blockSize - 1) / blockSize;
    CuSubset<<<numBlocks, blockSize>>>(
            weight_ptr,
            src_ptr,
            dest_ptr,
            phraseLength,
            embedDimension,
            vocabSize,
            batchSize);

    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    cudaDeviceSynchronize();
    return 0;

    Error:
    return 1;
}
} // extern C
