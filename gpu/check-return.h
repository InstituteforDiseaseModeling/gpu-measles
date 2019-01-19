#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Helper functions for CUDA runtime error checking

#define CheckReturn(err)    CheckError(err, __FUNCTION__, __LINE__)
#define CheckCurand(err)    CheckCurandError(err, __FUNCTION__, __LINE__)

inline void CheckError(cudaError_t const err, char const* const fun, const int line)
{
    if (err)
    {
        printf("CUDA Error Code[%d]: %s\n%s() Line:%d\n", err, cudaGetErrorString(err), fun, line);
        exit(1);
    }
}

inline void CheckCurandError(curandStatus_t const err, char const* const fun, const int line)
{
    if (err != CURAND_STATUS_SUCCESS)
    {
        printf("Curand error code[%d]: %s() Line: %d\n", err, fun, line);
        exit(1);
    }
}
