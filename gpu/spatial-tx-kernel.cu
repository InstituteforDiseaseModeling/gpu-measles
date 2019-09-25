#include "spatial-tx-kernel.h"
//#include "cub/cub/cub.cuh"
#include "cub/cub/block/block_reduce.cuh"

__global__ void total_population( uint32_t count, uint32_t* pPopulations, uint32_t* pTotal )
{
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint32_t population = ((tidx < count) ? pPopulations[tidx] : 0);

    typedef cub::BlockReduce<uint32_t, THREADS_PER_BLOCK> BlockReduceUint32;
    __shared__ typename BlockReduceUint32::TempStorage uint_storage;

    uint32_t total_population = BlockReduceUint32(uint_storage).Sum( population );

    __syncthreads();

    if ( tidx == 0 )
    {
        *pTotal = total_population;
    }
}

__global__ void calculate_contagion(
    uint32_t count,
    float* pFactorIJ,
    float* pInfectionForLocation,
    uint32_t* pPopulationForLocation,
    uint32_t* pTotalPopulation,
    float* pOutflows,
    float* pInflows,
    float  factor )
{
    uint32_t tidxx = threadIdx.x + (blockIdx.x * blockDim.x);   // destination node
    uint32_t tidxy = threadIdx.y + (blockIdx.y * blockDim.y);   // source node

    if ( ( tidxx < count ) && ( tidxy < count ) )
    {
        float* pFactor        = pFactorIJ              + (tidxy * count) + tidxx;
        float* pInfection     = pInfectionForLocation                    + tidxy;
        uint32_t* pPopulation = pPopulationForLocation                   + tidxy;
        float delta = factor * (*pInfection) * (*pPopulation) * (*pFactor) / (*pTotalPopulation);
        // printf("tidx (%d,%d) c=%f p=%d f=%f n=%d d=%f\n", tidxx, tidxy, *pInfection, *pPopulation, *pFactor, *pTotalPopulation, delta);
        float* pOutflow = pOutflows + tidxy;    // from Y
        float* pInflow  = pInflows  + tidxx;    // to X
        atomicAdd( pOutflow, delta );
        atomicAdd( pInflow,  delta );
    }
}

__global__ void update_contagion( uint32_t count, float* pInflow, float* pOutflow, float* pOutput )
{
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( tidx < count )
    {
        pOutput[ tidx ] += pInflow[ tidx ] - pOutflow[ tidx ];
    }
}
