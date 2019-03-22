/*
** copyright
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#define INFECTED    (0x01)
#define VACCINATED  (0x02)

struct Population {
    uint32_t capacity;
    uint32_t size;
    uint8_t* flags;                 // See #defines above
    uint8_t* elapsed_time;          // Up to 255 days, could need more for sum of exposure and infectious
    uint8_t* exposure_duration;     // Up to 255 days
    uint8_t* infectious_duration;   // Up to 255 days
    uint16_t* age_in_days;          // Up to 65535 days = 179.5 year
    uint16_t* location;             // Up to 65535 different locations
    uint16_t* immunity;             // 0 = susceptible, otherwise count-down "timer"
    __half* infectivity;            // Could be 0|1, might need floating point later

    Population( uint32_t capacity );
    ~Population();

    void LoadFile( const std::string& filename );
    void Update( size_t step, uint32_t* populations, float* contagion, OUTPUT current );

protected:
    curandState_t*  pPrngStates;
    uint16_t        numLocales;
    bool            isView;

    Population( const Population& source );

    void initializePrngStates();
    void loadJsonPopulationFile( const std::string& filename, std::vector<uint32_t>& populations );
    void instantiateNewIndividuals( size_t offset, uint32_t count, int32_t* new_individuals, OUTPUT stats );
    void scramblePopulation( size_t offset );
    void initializeNewIndividuals( size_t offset, uint32_t count, uint16_t mean_age, bool initial_immunity, OUTPUT stats );

    Population operator +(size_t offset);
};

