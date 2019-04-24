#pragma once

#include "population.h"
#include "measles-types.h"
#include "tx-log.h"

__global__ void update_population(
    uint16_t timeStep,
    Population state,
    Statistics current,
    Statistics output,
    TransmissionLog transmission
);
