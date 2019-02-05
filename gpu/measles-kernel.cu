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
        state.is_infected[tidx]       =    0;
        state.elapsed_time[tidx]      =    0;
        state.exposure_period[tidx]   =    0;
        state.infectious_period[tidx] =    0;
        state.recovery_period[tidx]   =    0;
        // state.age_in_days[tidx]       =    0;
        // state.location[tidx]          =    0;
        state.spatial_contagion[tidx] = 0xFF;
        state.immunity[tidx]          =    0;
        state.infectivity[tidx]       =    0;
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

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    float* force_of_infection,
    STATE state,
    OUTPUT output,
    curandState* prngState)
{
    int tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    bool is_infected = false;
    float immunity = 1.0f;  // If initialized to 0.0f, must set to 1.0f in if (!is_infected) else clause
    float infectivity = 0.0f;

    uint32_t susceptible = 0;
    uint32_t exposed     = 0;
    uint32_t infectious  = 0;
    uint32_t recovered   = 0;

    if ( tidx < agent_count ) {

        // (Even) children get older...
        ++state.age_in_days[tidx];

        is_infected = bool(state.is_infected[tidx]);
        if ( !is_infected ) {

            // May need to do this outside the is_infected check if move to MT prng.
            float draw = curand_uniform(&prngState[tidx]);
            immunity = float(state.immunity[tidx]);
            float susceptibility = force_of_infection[time_step] * (1.0f - immunity) / agent_count;

            if ( draw <= susceptibility ) {
                state.is_infected[tidx]       = uint8_t(true);
                state.elapsed_time[tidx]      = 0;
                state.exposure_period[tidx]   = randrange(&prngState[tidx], 2, 5);
                state.infectious_period[tidx] = randrange(&prngState[tidx], 5, 10);
                state.infectivity[tidx]       = 0.0f;
                state.recovery_period[tidx]   = (randrange(&prngState[tidx], 180, 361) - 180);
                state.immunity[tidx]          = 1.0f;
                state.spatial_contagion[tidx] = state.location[tidx];

                is_infected = true;
                immunity = 1.0f;
            }
        }
        else {
            uint32_t timer = state.elapsed_time[tidx] + 1;
            state.elapsed_time[tidx] = timer;
            uint32_t infectious_max = state.infectious_period[tidx];
            uint32_t infectious_time = min(max(timer - state.exposure_period[tidx], 0), infectious_max);
            if ( timer <= (state.exposure_period[tidx] + infectious_max) )
            {
                infectivity = max(1.0f - pow(2.0f * float(infectious_time) / infectious_max - 1, 2), 0.0f);
            }
            else
            {
                state.is_infected[tidx] = uint8_t(false);
                state.spatial_contagion[tidx] = 0xFFFF;
                infectivity = 0.0f; // may be unnecessary as infectivity should already be initialized to 0.0f
            }
            state.infectivity[tidx] = infectivity;
        }

        // bool is_infected  = state.is_infected[tidx];
        // float immunity    = float(state.immunity[tidx]);
        // float infectivity = float(state.infectivity[tidx]);
        susceptible = (!is_infected && (immunity    == 0.0f)) ? 1 : 0;
        exposed     = ( is_infected && (infectivity == 0.0f)) ? 1 : 0;
        infectious  = ( is_infected && (infectivity  > 0.0f)) ? 1 : 0;
        recovered   = (!is_infected && (immunity     > 0.0f)) ? 1 : 0;
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
