#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef struct {
    uint32_t* population;
    uint32_t* susceptible;
    uint32_t* exposed;
    uint32_t* infectious;
    uint32_t* recovered;
    uint32_t* incidence;
    float*    contagion;
} OUTPUT;
