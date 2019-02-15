#include <cmath>

#include "cub/cub/device/device_histogram.cuh" // #include <cub/cub.cuh>

#include "kernel.h"

__global__ void kInitializePrngs(uint32_t count, curandState* pPrngStates)
{
    uint32_t threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
    if ( threadId < count )
    {
        curand_init(threadId, 0, 0, &pPrngStates[threadId]);
    }
}

__global__ void kInitializeSamples(uint32_t count, float* pSamples, curandState* pPrngStates, DistributionType type, float a, float b)
{
    uint32_t threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
    if ( threadId < count )
    {
        float sample = 42;

        switch (type)
        {
            case UNIFORM:
                // a = min, b = max
                sample = a + curand_uniform(&pPrngStates[threadId]) * (b - a);
                break;

            case NORMAL:
                // a = mean, b = stddev
                sample = a + curand_normal(&pPrngStates[threadId]) * b;
                break;

            case POISSON:
                // a = mean, b unused
                sample = curand_poisson(&pPrngStates[threadId], a);
                break;

            case EXPONENTIAL:
                // a = mean, b unused
                sample = -log(curand_uniform(&pPrngStates[threadId])) / a;
                break;

            default:
                break;
        }

        pSamples[threadId] = sample;
    }
}
