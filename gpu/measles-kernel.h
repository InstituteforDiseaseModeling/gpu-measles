#pragma once

#include "measles-types.h"

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    uint32_t* populations,
    float* contagion,
    STATE state,
    OUTPUT output,
    curandState* prngState
);
