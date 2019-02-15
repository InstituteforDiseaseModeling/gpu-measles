#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

enum  DistributionType : uint32_t {
    UNIFORM = 0,
    NORMAL  = 1,
    POISSON = 2,
    EXPONENTIAL = 3
};

__global__ void kInitializePrngs(uint32_t count, curandState* pPrngStates);
__global__ void kInitializeSamples(uint32_t count, float* pSamples, curandState* pPrngStates, DistributionType type, float a, float b);
