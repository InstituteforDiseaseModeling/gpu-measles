#pragma once

#include <curand_kernel.h>

#include "population.h"
#include "statistics.h"

class VitalDynamics
{
public:

    VitalDynamics( float birth_rate, float mortality, uint32_t num_locations, curandState_t* prngStates, Population state, Statistics stats );
    ~VitalDynamics();

    uint32_t Update( uint32_t* populations, float dt, Statistics stats, uint32_t num_agents );

protected:

    float birthRate;            // crude birthrate (live births / 1000 people / time step)
    float nonDiseaseMortality;  // probability of death / time step
    uint32_t numLocations;
    int32_t* births;

    // Not owned by this object!
    curandState_t* pPrngStates;
    Population  state;
    Statistics  stats;

private:
};
