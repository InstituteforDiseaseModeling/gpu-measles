#pragma once

#include <cstdint>

#include "cuda_runtime.h"

// (uint32_t) incubation_duration minimum
// (uint32_t) incubation_duration maximum
// (uint32_t) infection_duration minimum
// (uint32_t) infection_duration maximum
// (float)    infectivity/time_step

// TODO - (float) sigmoid half_max
// TODO - (float) sigmoid steep_fac
// TODO - (float) sigmoid sus_init

struct Constants {
    uint32_t incubation_duration_minimum;
    uint32_t incubation_duration_maximum;
    uint32_t infection_duration_minimum;
    uint32_t infection_duration_maximum;
    float    infectivity;

    void UpdateDevice( void );
};
