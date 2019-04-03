/*
** copyright
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "census.h"
#include "statistics.h"

#define INFECTED    (0x01)
#define VACCINATED  (0x02)

struct Population {
    uint32_t size;
    uint8_t* flags;                 // See #defines above
    uint8_t* elapsed_time;          // Up to 255 days, could need more for sum of exposure and infectious
    uint8_t* exposure_duration;     // Up to 255 days
    uint8_t* infectious_duration;   // Up to 255 days
    uint16_t* age_in_days;          // Up to 65535 days = 179.5 year
    uint16_t* location;             // Up to 65535 different locations
    uint16_t* immunity;             // 0 = susceptible, otherwise count-down "timer"
    __half* infectivity;            // Could be 0|1, might need floating point later

    // Population();
    // Population( const Population& source );

    Population operator +(size_t offset);

// TODO - move this back to protected once VitalDynamics is part of PopulationImpl
//protected:
    curandState_t*  pPrngStates;
};

struct PopulationImpl : public Population
{
    uint32_t capacity;

    PopulationImpl( uint32_t capacity, uint32_t prng_seed );
    ~PopulationImpl();

    void Initialize( const Census& census );
    void Update( size_t step, uint32_t* populations, float* contagion, Statistics current );
    void VitalDynamics( float birth_rate, float mortality, float dt, Statistics current );

protected:
    uint16_t    numLocales;
    uint32_t*   populations;
    int32_t*    birthsByLocale;

    void initializePrngStates( uint32_t prng_seed );
    void instantiateNewIndividuals( size_t offset, uint32_t count, int32_t* new_individuals, Statistics stats );
    void scramblePopulation( size_t offset, uint32_t count );
    void initializeNewIndividuals( size_t offset, uint32_t count, uint16_t mean_age, bool initial_immunity, Statistics stats );
};
