#pragma once

#include "measles-types.h"

__global__ void initialize_prng(uint32_t count, curandState* prngState);

__global__ void zero_state(uint32_t count, STATE state);
__global__ void zero_output(uint32_t time_steps, uint32_t locations, OUTPUT output);

__global__ void initialize_ages(uint32_t count, STATE state, curandState* pPrngStates);
__global__ void initialize_locations(uint32_t count, uint32_t max_locations, STATE state, uint32_t* populations, curandState* pPrngStates);
__global__ void initialize_contagion(uint32_t count, uint32_t max_locations, uint32_t* populations, float* contagion);
