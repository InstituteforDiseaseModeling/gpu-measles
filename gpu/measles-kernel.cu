#include "measles-kernel.h"
#include "cub/cub/cub.cuh"
#include "block-size.h"

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

__global__ void zero_output(uint32_t time_steps, OUTPUT output)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < time_steps )
    {
        output.total_susceptible[tidx] = 0;
        output.total_exposed[tidx]     = 0;
        output.total_infectious[tidx]  = 0;
        output.total_recovered[tidx]   = 0;
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

__global__ void initialize_locations(uint32_t count, uint16_t max_locations, STATE state, curandState* pPrngStates)
{
    uint32_t tidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tidx < count )
    {
        uint16_t location = uint16_t(curand_uniform(&pPrngStates[tidx]) * max_locations);
        state.location[tidx] = location;
    }
}

__device__ uint32_t randrange(curandState* prngState, uint32_t min, uint32_t max);

__device__ inline void SusceptibleToInfected(uint32_t tidx, STATE state, curandState* pPrngState)
{
    state.is_infected[tidx]         = uint8_t(true);
    state.elapsed_time[tidx]        = 0;
    state.exposure_duration[tidx]   = randrange(pPrngState, 2, 5);
    state.infectious_duration[tidx] = randrange(pPrngState, 5, 10);
    state.infectivity[tidx]         = 0.0f;
    state.immunity[tidx]            = UINT16_MAX;
}

__device__ inline float UpdateInfected(uint32_t tidx, STATE state, bool& is_infected, bool& is_infectious)
{
    // is_infected = true;     // already set or we wouldn't be here
    // is_infectious = false;  // already set

    uint32_t timer               = ++state.elapsed_time[tidx];
    uint32_t exposure_duration   =   state.exposure_duration[tidx];
    uint32_t infectious_duration =   state.infectious_duration[tidx];
    // 0 <= infectious_timer <= infectious_duration
    uint32_t infectious_timer    = min(max(timer - exposure_duration, 0), infectious_duration);
    float infectivity = 0.0f;

    if ( timer >= exposure_duration )
    {
        if ( timer <= (exposure_duration + infectious_duration) )
        {
            // in infectious period
            is_infectious = true;
            infectivity = max(1.0f - pow(2.0f * float(infectious_timer) / infectious_duration - 1, 2), 0.0f);
            // TBD - infectivity might be zero based on this calculation, should
            // the individual still be considered infectious?
        }
        else
        {
            // recovered
            state.is_infected[tidx] = uint8_t(false);
            is_infected = false;
        }
    }
    else
    {
        // nothing, just progressing through exposure
    }

    state.infectivity[tidx] = infectivity;

    return infectivity;
}

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    float* force_of_infection,
    STATE state,
    OUTPUT output,
    curandState* prngState)
{
    int tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    uint32_t susceptible = 0;
    uint32_t exposed     = 0;
    uint32_t infectious  = 0;
    uint32_t recovered   = 0;

    float infectivity = 0.0f;

    if ( tidx < agent_count ) {

        bool is_infected   = bool(state.is_infected[tidx]);
        bool is_immune     = false;
        bool is_infectious = false;

        // (Even) children get older...
        ++state.age_in_days[tidx];

        if ( !is_infected )
        {
            is_immune = bool(state.immunity[tidx]);

            if ( !is_immune )
            {
                float draw = curand_uniform(&prngState[tidx]);
                float susceptibility = force_of_infection[time_step] / agent_count;

                if ( draw <= susceptibility )
                {
                    SusceptibleToInfected(tidx, state, &prngState[tidx]);
                    is_infected = true;
                }
            }
            else
            {
                --state.immunity[tidx];
            }
        }
        else
        {
            infectivity = UpdateInfected(tidx, state, is_infected, is_infectious);
        }

        susceptible = (!is_infected && !is_immune)     ? 1 : 0;
        exposed     = ( is_infected && !is_infectious) ? 1 : 0;
        infectious  = ( is_infected &&  is_infectious) ? 1 : 0;
        recovered   = (!is_infected &&  is_immune)     ? 1 : 0;
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_float_storage;

    float aggregate = BlockReduceFloat(temp_float_storage).Sum(infectivity);

    typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduceUint32;
    __shared__ typename BlockReduceUint32::TempStorage temp_uint_storage;

    uint32_t susceptible_count = BlockReduceUint32(temp_uint_storage).Sum(susceptible); __syncthreads();
    uint32_t exposed_count     = BlockReduceUint32(temp_uint_storage).Sum(exposed);     __syncthreads();
    uint32_t infectious_count  = BlockReduceUint32(temp_uint_storage).Sum(infectious);  __syncthreads();
    uint32_t recovered_count   = BlockReduceUint32(temp_uint_storage).Sum(recovered);

    if ( threadIdx.x == 0 )
    {
        // atomicAdd( ptr, value );
        atomicAdd(force_of_infection       + time_step + 1, aggregate);
        atomicAdd(output.total_susceptible + time_step,     susceptible_count);
        atomicAdd(output.total_exposed     + time_step,     exposed_count);
        atomicAdd(output.total_infectious  + time_step,     infectious_count);
        atomicAdd(output.total_recovered   + time_step,     recovered_count);
    }
}

// Returns an integer in the range [min, max). Note - no parameter checking. :)
__device__ uint32_t randrange(curandState* prngState, uint32_t min, uint32_t max) {
    uint32_t draw = curand(prngState);
    draw %= (max - min);
    draw += min;

    return draw;
}
