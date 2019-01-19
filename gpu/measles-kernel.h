#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void initialize_prng(curandState* prngState, uint32_t count);

typedef struct {
    uint32_t* is_infected;
    uint32_t* elapsed_time;
    uint32_t* exposure_period;
    uint32_t* infectious_period;
    uint32_t* recovery_period;
    float* immunity;
    float* infectivity;
} STATE;

typedef struct {
    uint32_t* total_susceptible;
    uint32_t* total_exposed;
    uint32_t* total_infectious;
    uint32_t* total_recovered;
} OUTPUT;

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    curandState* prngState,
    float* force_of_infection,
    STATE state,
    OUTPUT output
);
