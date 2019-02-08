#pragma once

#include "measles-types.h"

__global__ void vaccinate_const(
    uint32_t agent_count,
    uint32_t node,
    uint32_t min_age,
    uint32_t max_age,
    uint32_t duration,
    STATE state);

__global__ void vaccinate_poisson(
    uint32_t agent_count,
    uint32_t node,
    uint32_t min_age,
    uint32_t max_age,
    uint32_t duration,
    STATE state,
    curandState* pPrngState);
