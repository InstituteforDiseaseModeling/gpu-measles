#pragma once

#include "measles-types.h"

class VitalDynamics
{
public:

    VitalDynamics( float birth_rate, float mortality, uint32_t num_locations, curandState_t* prngStates, STATE state );
    ~VitalDynamics();

    uint32_t Update( uint32_t* populations, float dt, OUTPUT stats, uint32_t num_agents );

protected:

    float birthRate;            // crude birthrate (live births / 1000 people / time step)
    float nonDiseaseMortality;  // probability of death / time step
    uint32_t numLocations;
    int32_t* births;

    // Not owned by this object!
    curandState_t* pPrngStates;
    STATE   state;
    OUTPUT  stats;

private:
};
