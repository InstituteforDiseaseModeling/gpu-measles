/*

allocate for maximum population
keep track of current population
use x_Base_Population to scale current demographics
use birth rate and non-disease mortality to determine new agents
    - non-disease mortality age dependent?
    - maternal antibodies dependent on I/S ratio for LGA?
births dependent on birth rate and LGA population (Poisson?)
time period variable - daily/weekly/monthly/annually?
populate new agents at end of current population vector
shuffle new agents

*/

#include "vital-dynamics.h"

#include "cuda.h"
#include "block-size.h"

#include "cub/cub/cub.cuh"

/*
__global__ void vital_dynamics( uint32_t num_locations, uint32_t* populations, STATE state, uint32_t* indices, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        uint32_t my_location = UINT32_MAX;
        uint32_t running_total = 0;
        for ( uint32_t location = 0; location < num_locations; ++location )
        {
            running_total += populations[location];
            if ( running_total > tidx )
            {
                my_location = location;
                break;
            }
        }

        state.location[ tidx ] = my_location;
        indices[ tidx ] = curand( pPrngStates + tidx );
    }
}
*/

__device__ float sigmoid( curandState_t* pPrngState )
{
    float half_max  = 150;  // when 50% of individuals have lost their immunity, sample value for measles
    float steep_fac =  50;  // ~half-life of maternal antibodies, sample value for measles
    float sus_init  =   0;  // fraction of population susceptible at age = -∞, sample value for measles
    float draw = curand_uniform( pPrngState );
    float duration = half_max - steep_fac * log((1 - sus_init) / (draw - sus_init) - 1 + 0.001);
    return max(duration, 0.0f); // clamp return value to 0
}

// How many people are born in each spatial location?
// population * rate * Δt
__global__ void births_for_time_period( uint32_t* populations, uint32_t num_locations, float rate, float dt, int32_t* births, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < num_locations )
    {
        births[ tidx ] = curand_poisson( pPrngStates + tidx, populations[ tidx ] * rate * dt );
    }
}

// Non-disease deaths (all become new infants assuming birthrate > mortality rate)
__global__ void non_disease_deaths( STATE state, OUTPUT stats, uint32_t count, float probability, int32_t* births, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        float draw = curand_uniform( pPrngStates + tidx );
        if ( draw <= probability )
        {
            // printf("%09d, aged %d, died\n", tidx, uint32_t(state.age_in_days[ tidx ]) );

            state.flags[ tidx ]               = 0;
            state.elapsed_time[ tidx ]        = 0;
            state.exposure_duration[ tidx ]   = 0;
            state.infectious_duration[ tidx ] = 0;
            state.age_in_days[ tidx ]         = 0;
            uint32_t location = state.location[ tidx ];  // Use current location, i.e. non-disease deaths -> births
            // immunity depends on S/(S+R)?
            float draw = curand_uniform( pPrngStates + tidx );
            float susceptible = float(stats.susceptible[ location ]);
            float recovered   = float(stats.recovered[ location ]);
            if ( draw < (recovered / (susceptible + recovered)) )
            {
                state.immunity[ tidx ] = uint32_t( sigmoid( pPrngStates + tidx ) );
            }
            else
            {
                state.immunity[ tidx ] = 0;
            }
            state.infectivity[ tidx ] = 0;

            --births[ location ];
        }
    }
}

// Additional births
__global__ void new_births( uint32_t count, uint32_t num_locations, int32_t* births, STATE state, uint16_t* indices, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        uint32_t my_location = UINT32_MAX;
        uint32_t running_total = 0;
        for ( uint32_t location = 0; location < num_locations; ++location )
        {
            running_total += births[ location ];
            if ( running_total >= tidx )
            {
                my_location = location;
                break;
            }
        }

        // state.location[ tidx ] = my_location;
        // indices[ tidx ] = curand( pPrngStates + tidx ) % num_locations;
        // Instead of allocating and de-allocating memory, we will temporarily use age_in_days and immunity for storage.
        state.age_in_days[ tidx ] = my_location;
        state.immunity[ tidx ]    = uint16_t(curand( pPrngStates + tidx ) % 0xFFFF);
    }
}

// shuffle new births (randomizes atomic accesses)
__global__ void shuffle_agents()
{

}

__global__ void initialize_newborns( uint32_t count, STATE state, OUTPUT stats, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        state.flags[ tidx ]               = 0;
        state.elapsed_time[ tidx ]        = 0;
        state.exposure_duration[ tidx ]   = 0;
        state.infectious_duration[ tidx ] = 0;
        state.age_in_days[ tidx ]         = 0;
        uint32_t location = state.location[ tidx ];
        float draw = curand_uniform( pPrngStates + tidx );
        float susceptible = stats.susceptible[ location ];
        float recovered   = stats.recovered[ location ];
        if ( draw < (recovered / (susceptible + recovered)) )
        {
            state.immunity[ tidx ] = uint32_t( sigmoid( pPrngStates + tidx ) );
        }
        else
        {
            state.immunity[ tidx ] = 0;
        }
        state.infectivity[ tidx ] = 0;
    }
}

/******************************************************************************/

VitalDynamics::VitalDynamics( float birth_rate, float mortality, uint32_t num_locations, curandState_t* prngStates, STATE state )
    : birthRate( birth_rate )
    , nonDiseaseMortality( mortality )
    , numLocations( num_locations )
    , births( nullptr )
    , pPrngStates( prngStates )
    , state( state )
{
    cuda::mallocManaged( &births, numLocations * sizeof(births[0]) );
}

VitalDynamics::~VitalDynamics()
{
    cuda::free( births );
    births = nullptr;
}

uint32_t VitalDynamics::Update( uint32_t* populations, float dt, OUTPUT stats, uint32_t num_agents )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (numLocations + blocks.x - 1) / blocks.x, 1, 1 );

    births_for_time_period<<<grid, blocks>>>( populations, numLocations, birthRate, dt, births, pPrngStates );
    cuda::deviceSynchronize();

    grid = dim3( (num_agents + blocks.x - 1) / blocks.x, 1, 1 );
    non_disease_deaths<<<grid, blocks>>>( state, stats, num_agents, nonDiseaseMortality, births, pPrngStates );
    cuda::deviceSynchronize();

    uint32_t new_individuals = 0;
    for ( uint32_t iLocation = 0; iLocation < numLocations; ++iLocation )
    {
        uint32_t count = births[ iLocation ];
        if ( count >= 0 )
        {
            new_individuals += count;
        }
        else
        {
            births[ iLocation ] = 0;
        }
    }

    grid = dim3( (new_individuals + blocks.x - 1) / blocks.x, 1, 1 );
    STATE newState = state;
    newState.flags               += num_agents;
    newState.elapsed_time        += num_agents;
    newState.exposure_duration   += num_agents;
    newState.infectious_duration += num_agents;
    newState.age_in_days         += num_agents;
    newState.location            += num_agents;
    newState.immunity            += num_agents;
    newState.infectivity         += num_agents;
    new_births<<<grid, blocks>>>( new_individuals, numLocations, births, newState, newState.immunity, pPrngStates );
    cuda::deviceSynchronize();

    void* scratch = nullptr;
    size_t scratch_bytes;
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, new_individuals );
    cuda::malloc( &scratch, scratch_bytes );
    // Keys (random draws) are in immunity, use infectivity for output.
    // Locations are, initially, in age_in_days, store into "location".
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, newState.immunity, (uint16_t*)newState.infectivity, newState.age_in_days, newState.location, new_individuals );
    cuda::deviceSynchronize();
    cuda::free( scratch );

    initialize_newborns<<<grid, blocks>>>( new_individuals, newState, stats, pPrngStates );
    cuda::deviceSynchronize();

    return new_individuals;
}