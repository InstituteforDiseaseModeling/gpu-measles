#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

typedef struct {
    uint8_t* is_infected;           // Generally 0|1, could track up to eight strains with bits
    uint8_t* elapsed_time;          // Up to 255 days, could need more for sum of exposure and infectious
    uint8_t* exposure_duration;     // Up to 255 days
    uint8_t* infectious_duration;   // Up to 255 days
    uint16_t* age_in_days;          // Up to 65535 days = 179.5 year
    uint16_t* location;             // Up to 65535 different locations
    uint16_t* immunity;             // 0 = susceptible, otherwise count-down "timer"
    __half* infectivity;            // Could be 0|1, might need floating point later
} STATE;

typedef struct {
    uint32_t* population;
    uint32_t* susceptible;
    uint32_t* exposed;
    uint32_t* infectious;
    uint32_t* recovered;
    uint32_t* incidence;
    float*    contagion;
} OUTPUT;
