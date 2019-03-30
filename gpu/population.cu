/*
** copyright
*/

#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "block-size.h"
#include "census.h"
#include "cuda.h"
#include "demographics.h"
#include "measles-kernel.h"
#include "population.h"
#include "utilities.h"

#include "cub/cub/cub.cuh"

Population Population::operator +(size_t offset)
{
    Population result( *this );
    result.flags               += offset;
    result.elapsed_time        += offset;
    result.exposure_duration   += offset;
    result.infectious_duration += offset;
    result.age_in_days         += offset;
    result.location            += offset;
    result.immunity            += offset;
    result.infectivity         += offset;
    result.pPrngStates         += offset;

    return result;
}

PopulationImpl::PopulationImpl( uint32_t capacity, uint32_t prng_seed )
    : capacity( capacity )
    , numLocales( 0 )
    , populations( nullptr )
{
    // Using sizeof(field[0]) lets us change the datatypes more easily...

    cuda::mallocManaged( &flags,               capacity * sizeof(flags[0]) );
    cuda::mallocManaged( &elapsed_time,        capacity * sizeof(elapsed_time[0]) );
    cuda::mallocManaged( &exposure_duration,   capacity * sizeof(exposure_duration[0]) );
    cuda::mallocManaged( &infectious_duration, capacity * sizeof(infectious_duration[0]) );
    cuda::mallocManaged( &age_in_days,         capacity * sizeof(age_in_days[0]) );
    cuda::mallocManaged( &location,            capacity * sizeof(location[0]) );
    cuda::mallocManaged( &immunity,            capacity * sizeof(immunity[0]) );
    cuda::mallocManaged( &infectivity,         capacity * sizeof(infectivity[0]) );

    cuda::memset( flags,               0, capacity * sizeof(flags[0]) );
    cuda::memset( elapsed_time,        0, capacity * sizeof(elapsed_time[0]) );
    cuda::memset( exposure_duration,   0, capacity * sizeof(exposure_duration[0]) );
    cuda::memset( infectious_duration, 0, capacity * sizeof(infectious_duration[0]) );
    cuda::memset( age_in_days,         0, capacity * sizeof(age_in_days[0]) );
    cuda::memset( location,            0, capacity * sizeof(location[0]) );
    cuda::memset( immunity,            0, capacity * sizeof(immunity[0]) );
    cuda::memset( infectivity,         0, capacity * sizeof(infectivity[0]) );

    cuda::mallocManaged( &pPrngStates, capacity * sizeof(curandState) );
    initializePrngStates( prng_seed );

    cuda::deviceSynchronize();
}

PopulationImpl::~PopulationImpl()
{
    cuda::free( populations );
    cuda::free( pPrngStates );
    cuda::free( infectivity );
    cuda::free( immunity );
    cuda::free( location );
    cuda::free( age_in_days );
    cuda::free( infectious_duration );
    cuda::free( exposure_duration );
    cuda::free( elapsed_time );
    cuda::free( flags );
}

__global__ void initialize_prng(uint32_t count, uint32_t seed, curandState* prngState)
{
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( tidx < count ) {
        // curand_init( tidx, 0, 0, &prngState[ tidx ]);
        curand_init( seed* tidx, 0, 0, &prngState[ tidx ]);
    }
}

void PopulationImpl::initializePrngStates( uint32_t prng_seed )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (capacity + blocks.x - 1) / blocks.x, 1, 1 );

    initialize_prng<<<grid, blocks>>>( capacity, prng_seed, pPrngStates );
    cuda::deviceSynchronize();
}

void PopulationImpl::Initialize( const Census& census )
{
    numLocales = census.population.size();
    size = std::accumulate( census.population.begin(), census.population.end(), 0 );

    cuda::mallocManaged( &populations, numLocales * sizeof(populations[0]) );
    cuda::memcpy( populations, census.population.data(), numLocales * sizeof(populations[0]), cudaMemcpyDefault );

    Statistics stats;
    instantiateNewIndividuals( 0, size, reinterpret_cast<int32_t*>(populations), stats );
}

__global__ void new_individuals( uint32_t count, uint32_t num_locations, int32_t* births, Population state, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        uint32_t my_location = UINT32_MAX;
        uint32_t running_total = 0;
        for ( uint32_t location = 0; location < num_locations; ++location )
        {
            running_total += births[ location ];
            if ( running_total > tidx )
            {
                my_location = location;
                break;
            }
        }

        // Instead of allocating and de-allocating memory, we will temporarily use age_in_days and immunity for storage.
        state.age_in_days[ tidx ] = my_location;
        state.immunity[ tidx ]    = uint16_t(curand( pPrngStates + tidx ) % 0xFFFF);
    }
}

#define MEAN_AGE    ( 25 * 365)

void PopulationImpl::instantiateNewIndividuals( size_t offset, uint32_t count, int32_t* individuals, Statistics stats )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (size + blocks.x - 1) / blocks.x, 1, 1 );
    Population temp = *this + offset;
    new_individuals<<<grid, blocks>>>( size, numLocales, individuals, temp, pPrngStates );
    cuda::deviceSynchronize();

    scramblePopulation( offset, count );
    initializeNewIndividuals( 0, size, MEAN_AGE, false, stats );
}

void PopulationImpl::scramblePopulation( size_t offset, uint32_t count )
{
    void* scratch = nullptr;
    size_t scratch_bytes;
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, count );
    cuda::malloc( &scratch, scratch_bytes );
    Population temp = *this + offset;
    // Keys (random draws) are in immunity, use infectivity for output.
    // Locations are, initially, in age_in_days, store into "location".
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, temp.immunity, (uint16_t*)temp.infectivity, temp.age_in_days, temp.location, count );
    cuda::deviceSynchronize();
    cuda::free( scratch );
}

#define MAX_AGE     (100 * 365)

__device__ float sigmoid( curandState_t* pPrngState )
{
    float half_max  = 150;  // when 50% of individuals have lost their immunity, sample value for measles
    float steep_fac =  50;  // ~half-life of maternal antibodies, sample value for measles
    float sus_init  =   0;  // fraction of population susceptible at age = -∞, sample value for measles
    float draw = curand_uniform( pPrngState );
    float duration = half_max - steep_fac * log((1 - sus_init) / (draw - sus_init) - 1 + 0.001);
    return max(duration, 0.0f); // clamp return value to 0
}

__global__ void initialize_newborns( uint32_t count, Population state, uint16_t mean_age, bool initial_immunity, Statistics stats, curandState_t* pPrngStates )
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        state.flags[ tidx ]               = 0;
        state.elapsed_time[ tidx ]        = 0;
        state.exposure_duration[ tidx ]   = 0;
        state.infectious_duration[ tidx ] = 0;
        uint32_t age = (mean_age > 0) ? uint32_t(-log(curand_uniform(&pPrngStates[tidx])) * mean_age) : 0;
        state.age_in_days[tidx] = min(age, MAX_AGE);
        uint32_t location = state.location[ tidx ];
        if ( initial_immunity )
        {
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
        }
        else
        {
            state.immunity[ tidx ] = 0;
        }
        state.infectivity[ tidx ] = 0;
    }
}

void PopulationImpl::initializeNewIndividuals( size_t offset, uint32_t count, uint16_t mean_age, bool initial_immunity, Statistics stats )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (count + blocks.x - 1) / blocks.x, 1, 1 );
    Population temp = *this + offset;
    Statistics empty;
    initialize_newborns<<<grid, blocks>>>( count, temp, mean_age, initial_immunity, empty, pPrngStates );
    cuda::deviceSynchronize();
}

void PopulationImpl::Update( size_t time_step, uint32_t* populations, float* contagion, Statistics current )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (size + blocks.x - 1) / blocks.x, 1, 1 );
    do_step<<<grid, blocks>>>(
        size,
        time_step,
        populations,
        contagion,
        *this,
        current,
        pPrngStates);

    cuda::deviceSynchronize();
}

/*
void DisplayAges(uint32_t agent_count, Population& state)
{
    if ( !debug) return;

    std::cout << "Age (in days):" << std::endl;
    for (uint32_t i = 0; i < min(32, agent_count); ++i)
    {
        std::cout << std::setw(2) << i << ": " << state.age_in_days[i] << std::endl;
    }
    std::cout << std::endl;
}

void DisplayLocations(uint32_t agent_count, Population& state)
{
    if ( !debug ) return;

    std::cout << "Location (ID):" << std::endl;
    for (uint32_t i = 0; i < min(32, agent_count); ++i)
    {
        std::cout << std::setw(2) << i << ": " << state.location[i] << std::endl;
    }
    std::cout << std::endl;
}
*/
