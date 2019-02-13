#include "measles-init.h"

__global__ void initialize_prng(uint32_t count, curandState* prngState)
{
    uint32_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if ( i < count ) {
        curand_init(i, 0, 0, &prngState[i]);
    }
}

#define MEAN_AGE    ( 25 * 365)
#define MAX_AGE     (100 * 365)

__global__ void initialize_ages(uint32_t count, STATE state, curandState* pPrngStates)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        uint32_t age = uint32_t(-log(curand_uniform(&pPrngStates[tidx])) * MEAN_AGE);
        state.age_in_days[tidx] = min(age, MAX_AGE);
    }
}

__global__ void initialize_locations(
    uint32_t count,
    uint32_t max_locations,
    uint32_t* populations,
    STATE state)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        uint32_t my_location = UINT32_MAX;
        uint32_t running_total = 0;
        for ( uint32_t location = 0; location < max_locations; ++location )
        {
            running_total += populations[location];
            if ( running_total > tidx )
            {
                my_location = location;
                break;
            }
        }

        state.location[tidx] = my_location;
    }
}

