#pragma once

#include "population.h"
#include "measles-types.h"

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    uint32_t* populations,
    float* contagion,
    Population state,
    Statistics output,
    curandState* prngState
);
