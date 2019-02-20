#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "check-return.h"

namespace cuda
{
    inline void deviceSynchronize( void ) { CheckReturn(cudaDeviceSynchronize()); CheckReturn(cudaGetLastError()); }
    inline void malloc( void* pPointer, size_t size ) { CheckReturn(cudaMalloc( (void**)pPointer, size )); CheckReturn(cudaGetLastError()); }
    inline void mallocManaged( void* pPointer, size_t size ) { CheckReturn(cudaMallocManaged( (void**)pPointer, size )); CheckReturn(cudaGetLastError()); }
    inline void memset( void* pointer, uint32_t value, size_t size ) { CheckReturn(cudaMemset( pointer, value, size )); CheckReturn(cudaGetLastError()); }
    inline void memcpy( void* destination, const void* source, size_t size, cudaMemcpyKind kind ) { CheckReturn(cudaMemcpy(destination, source, size, kind)); CheckReturn(cudaGetLastError()); }
}