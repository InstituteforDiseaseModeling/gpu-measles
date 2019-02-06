#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

__global__ void initialize_prng(uint32_t count, curandState* prngState);

typedef struct {
    uint8_t* is_infected;           // Generally 0|1, could track up to eight strains with bits
    uint8_t* elapsed_time;          // Up to 255 days, could need more for sum of exposure and infectious
    uint8_t* exposure_duration;     // Up to 255 days
    uint8_t* infectious_duration;   // Up to 255 days
    uint16_t* age_in_days;          // Up to 65535 days = 179.5 year
    uint16_t* location;             // Up to 65535 different locations
    uint16_t* immunity;             // 0 = susceptible, otherwise count-down "timer"
    __half* infectivity;            // Could be 0|1, might need floating point later
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
