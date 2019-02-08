#include "measles-init.h"

__global__ void initialize_prng(uint32_t count, curandState* prngState)
{
    uint32_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if ( i < count ) {
        curand_init(i, 0, 0, &prngState[i]);
    }
}

__global__ void zero_state(uint32_t count, STATE state)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        state.is_infected[tidx]         = 0;
        state.elapsed_time[tidx]        = 0;
        state.exposure_duration[tidx]   = 0;
        state.infectious_duration[tidx] = 0;
        // state.age_in_days[tidx]         = 0;
        // state.location[tidx]            = 0;
        state.immunity[tidx]            = 0;
        state.infectivity[tidx]         = 0;
    }
}

__global__ void zero_output(uint32_t time_steps, uint32_t locations, OUTPUT output)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < time_steps )
    {
        memset( output.population  + tidx * locations, 0, locations * sizeof(output.population[0]) );
        memset( output.susceptible + tidx * locations, 0, locations * sizeof(output.susceptible[0]) );
        memset( output.exposed     + tidx * locations, 0, locations * sizeof(output.exposed[0]) );
        memset( output.infectious  + tidx * locations, 0, locations * sizeof(output.infectious[0]) );
        memset( output.recovered   + tidx * locations, 0, locations * sizeof(output.recovered[0]) );
        memset( output.incidence   + tidx * locations, 0, locations * sizeof(output.incidence[0]) );
        memset( output.contagion   + tidx * locations, 0, locations * sizeof(output.contagion[0]) );
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

