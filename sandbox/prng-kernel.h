#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void initialize_prng(curandState* pPrngState, uint32_t count);

__global__ void uniform_default(curandState* pPrngState, float* pVariates, uint32_t count);
__global__ void gaussian_default(curandState* pPrngState, float* pVariates, uint32_t count);
__global__ void exponential_default(curandState* pPrngState, float mean, float* pVariates, uint32_t count);
__global__ void poisson_default(curandState* pPrngState, float lambda, float* pVariates, uint32_t count);

__global__ void uniform_mtgp32(curandStateMtgp32* pMTGP32States, float* pVariates, uint32_t count);
__global__ void gaussian_mtgp32(curandStateMtgp32* pMTGP32States, float* pVariates, uint32_t count);
__global__ void exponential_mtgp32(curandStateMtgp32* pMTGP32States, float mean, float* pVariates, uint32_t count);
__global__ void poisson_mtgp32(curandStateMtgp32* pMTGP32States, float lambda, float* pVariates, uint32_t count);
