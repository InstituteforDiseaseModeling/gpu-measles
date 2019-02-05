#include "prng-kernel.h"
#include "block-size.h"

__global__ void initialize_prng(curandState* pPrngState, uint32_t count)
{
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < count)
    {
        curand_init(i, 0, 0, &pPrngState[i]);
    }
}

__global__ void uniform_default(curandState* pPrngState, float* pVariates, uint32_t count)
{
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < count)
    {
        pVariates[i] = curand_uniform(&pPrngState[i]);
    }
}

__global__ void gaussian_default(curandState* pPrngState, float* pVariates, uint32_t count)
{
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < count)
    {
        pVariates[i] = curand_normal(&pPrngState[i]);
    }
}

__global__ void exponential_default(curandState* pPrngState, float mean, float* pVariates, uint32_t count)
{
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < count)
    {
        pVariates[i] = -log(curand_uniform(&pPrngState[i])) * mean;
    }
}

__global__ void poisson_default(curandState* pPrngState, float lambda, float* pVariates, uint32_t count)
{
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < count)
    {
        pVariates[i] = float(curand_poisson(&pPrngState[i], lambda));
    }
}

__global__ void uniform_mtgp32(curandStateMtgp32* pMTGP32States, float* pVariates, uint32_t count)
{
    uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t prng = thread / 256;

    if (thread < count)
    {
        pVariates[thread] = curand_uniform(&pMTGP32States[prng]);
    }
}

__global__ void gaussian_mtgp32(curandStateMtgp32* pMTGP32States, float* pVariates, uint32_t count)
{
    uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t prng = thread / 256;

    if (thread < count)
    {
        pVariates[thread] = curand_normal(&pMTGP32States[prng]);
    }
}

__global__ void exponential_mtgp32(curandStateMtgp32* pMTGP32States, float mean, float* pVariates, uint32_t count)
{
    uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t prng = thread / 256;

    if (thread < count)
    {
        pVariates[thread] = -log(curand_uniform(&pMTGP32States[prng])) * mean;
    }
}

__global__ void poisson_mtgp32(curandStateMtgp32* pMTGP32States, float lambda, float* pVariates, uint32_t count)
{
    uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t prng = thread / 256;

    if (thread < count)
    {
        pVariates[thread] = float(curand_poisson(&pMTGP32States[prng], lambda));
    }
}
