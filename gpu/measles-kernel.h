#pragma once

#include "population.h"
#include "measles-types.h"

__global__ void update_population(
    Population state,
    Statistics current,
    Statistics output
);
