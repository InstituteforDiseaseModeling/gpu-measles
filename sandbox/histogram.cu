#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cub/cub/cub.cuh"

#include "../gpu/check-return.h"
#include "kernel.h"

void AllocateMemory(uint32_t count, float** ppSamples, curandState** ppPrngStates, uint32_t** ppHistogram);
void InitializePrngs(uint32_t count, curandState* pPrngStates);
void InitializeSamples(uint32_t count, float* pSamples, curandState* pPrngStates);
void DisplaySamples(uint32_t count, float* pSamples);
void CalculateHistogram(uint32_t count, float* pSamples, uint32_t* pHistogram);
void DisplayResults(uint32_t* pHistogram);

int main(int argc, const char* argv[])
{
    uint32_t count = 1024;
    float* pSamples = nullptr;
    curandState* pPrngStates = nullptr;
    uint32_t* pHistogram;

    AllocateMemory(count, &pSamples, &pPrngStates, &pHistogram);
    InitializePrngs(count, pPrngStates);
    InitializeSamples(count, pSamples, pPrngStates);
    DisplaySamples(count, pSamples);
    CalculateHistogram(count, pSamples, pHistogram);
    DisplayResults(pHistogram);

    return 0;
}

#define NUM_BINS    (32)

void AllocateMemory(uint32_t count, float** ppSamples, curandState** ppPrngStates, uint32_t** ppHistogram)
{
    CheckReturn(cudaMallocManaged(ppSamples, count * sizeof(float)));
    CheckReturn(cudaMalloc(ppPrngStates, count * sizeof(curandState)));
    CheckReturn(cudaMallocManaged(ppHistogram, NUM_BINS * sizeof(uint32_t)));
}

#define THREADS_PER_BLOCK  (256)

void InitializePrngs(uint32_t count, curandState* pPrngStates)
{
    dim3 blocks(THREADS_PER_BLOCK);
    dim3 grid((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    kInitializePrngs<<<grid, blocks>>>(count, pPrngStates);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
}

void InitializeSamples(uint32_t count, float* pSamples, curandState* pPrngStates)
{
    dim3 blocks(THREADS_PER_BLOCK);
    dim3 grid((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    // kInitializeSamples<<<grid, blocks>>>(count, pSamples, pPrngStates, UNIFORM, 0, 32);
    // kInitializeSamples<<<grid, blocks>>>(count, pSamples, pPrngStates, NORMAL, 16, 4);
    // kInitializeSamples<<<grid, blocks>>>(count, pSamples, pPrngStates, POISSON, 8, 0);
    kInitializeSamples<<<grid, blocks>>>(count, pSamples, pPrngStates, EXPONENTIAL, 0.125, 0);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
}

void DisplaySamples(uint32_t count, float* pSamples)
{
    std::cout << "Samples: " << std::endl;
    uint32_t end = min(count, 32);
    for (uint32_t i = 0; i < end; ++i)
    {
        std::cout << std::setw(2) << i << ": " << pSamples[i] << std::endl;
    }
    std::cout << std::endl;
}

void CalculateHistogram(uint32_t count, float* pSamples, uint32_t* pHistogram)
{
    void*  storage = nullptr;
    size_t bytes   = 0;

    // storage == nullptr means required bytes will be returned in bytes (passed by reference)
    cub::DeviceHistogram::HistogramEven( storage, bytes, (float*)nullptr,  (int32_t*)nullptr,    NUM_BINS, 0.0f, float(NUM_BINS), int(count) );
    printf( "Allocating %zu bytes for histogram of %d items.\n", bytes, count );
    CheckReturn( cudaMalloc( &storage, bytes ) );
    cub::DeviceHistogram::HistogramEven( storage, bytes,         pSamples, (int32_t*)pHistogram, NUM_BINS, 0.0f, float(NUM_BINS), int(count) );
    cudaFree( storage );
}

void DisplayResults(uint32_t* pHistogram)
{
    std::cout << "Histogram: " << std::endl;
    uint32_t width = uint32_t(log10(NUM_BINS) + 1);
    for (uint32_t i = 0; i < NUM_BINS; ++i)
    {
        std::cout << std::setw(width) << i << ", " << pHistogram[i] << std::endl;
    }
    std::cout << std::endl;
}
