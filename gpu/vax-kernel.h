/*
** copyright
*/

#pragma once

#include <curand_kernel.h>

#include "population.h"

__global__ void vaccinate_const(
    uint32_t agent_count,
    uint32_t node,
    uint32_t min_age,
    uint32_t max_age,
    uint32_t duration,
    Population state);

__global__ void vaccinate_poisson(
    uint32_t agent_count,
    uint32_t node,
    uint32_t min_age,
    uint32_t max_age,
    uint32_t duration,
    Population state,
    curandState* pPrngState);
