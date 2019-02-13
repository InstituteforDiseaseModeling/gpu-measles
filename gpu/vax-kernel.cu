#include "vax-kernel.h"

__global__ void vaccinate_const(
    uint32_t agent_count,
    uint32_t node,
    uint32_t min_age,
    uint32_t max_age,
    uint32_t duration,
    STATE state)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < agent_count )
    {
        uint32_t location = state.location[ tidx ];
        uint32_t age      = state.age_in_days[ tidx ];
        bool     infected = bool(state.flags[ tidx ] & INFECTED);

        if ( (node == location) &&
             ( age >= min_age) && (age < max_age) &&
             !infected )
        {
            state.immunity[ tidx ] = duration;
        }
    }
}

__global__ void vaccinate_poisson(
    uint32_t agent_count,
    uint32_t node,
    uint32_t min_age,
    uint32_t max_age,
    uint32_t duration,
    STATE state,
    curandState* pPrngState)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < agent_count )
    {
        uint32_t location = state.location[ tidx ];
        uint32_t age      = state.age_in_days[ tidx ];
        bool     infected = bool(state.flags[ tidx ] & INFECTED);

        if ( (node == location) &&
             ( age >= min_age) && (age < max_age) &&
             !infected )
        {
            state.immunity[ tidx ] = curand_poisson(pPrngState + tidx, duration);
        }
    }
}