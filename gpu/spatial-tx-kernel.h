#pragma once

#include <cstdint>

#define THREADS_PER_BLOCK   (32)

__global__ void total_population( uint32_t count, uint32_t* pPopulations, uint32_t* pTotal );
__global__ void calculate_contagion(
    uint32_t count,
    float* pFactorIJ,
    float* pInfectionForLocation,
    uint32_t* pPopulationForLocation,
    uint32_t* pTotalPopulation,
    float* pOutflows,
    float* pInflows,
    float  factor );
__global__ void update_contagion( uint32_t count, float* pInflow, float* pOutflow, float*pOutput );
