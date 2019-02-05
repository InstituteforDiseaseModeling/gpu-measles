#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

__global__ void initialize_prng(uint32_t count, curandState* prngState);

typedef struct {
    uint8_t* is_infected;
    uint8_t* elapsed_time;
    uint8_t* exposure_period;
    uint8_t* infectious_period;
    uint8_t* recovery_period;
    uint16_t* age_in_days;
    uint16_t* location;
    uint16_t* spatial_contagion;
    __half* immunity;
    __half* infectivity;
} STATE;

typedef struct {
    uint32_t* total_susceptible;
    uint32_t* total_exposed;
    uint32_t* total_infectious;
    uint32_t* total_recovered;
} OUTPUT;

__global__ void zero_state(uint32_t count, STATE state);
__global__ void zero_output(uint32_t time_steps, OUTPUT output);

__global__ void initialize_ages(uint32_t count, STATE state, curandState* pPrngStates);
__global__ void initialize_locations(uint32_t count, uint16_t max_locations, STATE state, curandState* pPrngStates);

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    float* force_of_infection,
    STATE state,
    OUTPUT output,
    curandState* prngState
);
