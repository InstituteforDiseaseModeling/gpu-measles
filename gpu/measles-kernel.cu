#include "measles-kernel.h"
#include "../cub-1.8.0/cub/cub.cuh"
#include "block-size.h"

__global__ void initialize_prng(curandState* prngState, uint32_t count) {
    uint32_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if ( i < count ) {
        curand_init(i, 0, 0, &prngState[i]);
    }
}

__device__ uint32_t randrange(curandState* prngState, uint32_t min, uint32_t max);

__global__ void do_step(
    uint32_t agent_count,
    uint32_t time_step,
    curandState* prngState,
    float* force_of_infection,
    STATE state,
    OUTPUT output)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if ( i < agent_count ) {
        float draw = curand(&prngState[i]) / float(UINT_MAX);
        float susceptibility = force_of_infection[time_step] * (1.0f - state.immunity[i]) / agent_count;

        if ( !bool(state.is_infected[i]) ) {
            if ( draw <= susceptibility ) {
                state.is_infected[i]       = uint32_t(true);
                state.elapsed_time[i]      = 0;
                state.exposure_period[i]   = randrange(&prngState[i], 2, 5);
                state.infectious_period[i] = randrange(&prngState[i], 5, 10);
                state.infectivity[i]       = 0.0f;
                state.recovery_period[i]   = (randrange(&prngState[i], 180, 361) - 180);
                state.immunity[i]          = 1.0f;
            }
        }
        else {
            uint32_t timer = state.elapsed_time[i] + 1;
            state.elapsed_time[i] = timer;
            uint32_t infectious_max = state.infectious_period[i];
            uint32_t infectious_time = min(max(timer - state.exposure_period[i], 0), infectious_max);
            state.infectivity[i] = max(1.0f - pow(2.0f * float(infectious_time) / infectious_max - 1, 2), 0.0f);
            if ( timer > (state.exposure_period[i] + infectious_max) )
                state.is_infected[i] = uint32_t(false);
        }
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_float_storage;
    float aggregate = BlockReduceFloat(temp_float_storage).Sum((i < agent_count) ? state.infectivity[i] : 0.0f);
    typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduceUint32;
    __shared__ typename BlockReduceUint32::TempStorage temp_uint_storage;
    uint32_t susceptible_count = BlockReduceUint32(temp_uint_storage).Sum((i < agent_count) && !state.is_infected[i] && (state.immunity[i] == 0.0f) ? 1 : 0);
    __syncthreads();
    uint32_t exposed_count = BlockReduceUint32(temp_uint_storage).Sum((i < agent_count) && state.is_infected[i] && (state.infectivity[i] == 0.0f) ? 1 : 0);
    __syncthreads();
    uint32_t infectious_count = BlockReduceUint32(temp_uint_storage).Sum(((i < agent_count) && state.is_infected[i] && (state.infectivity[i] > 0.0f)) ? 1 : 0);
    __syncthreads();
    uint32_t recovered_count = BlockReduceUint32(temp_uint_storage).Sum((i < agent_count) && !state.is_infected[i] && (state.immunity[i] > 0.0f) ? 1 : 0);
    if ( threadIdx.x == 0 )
    {
        // atomicAdd( ptr, value );
        atomicAdd(force_of_infection + time_step + 1,   aggregate);
        atomicAdd(output.total_susceptible + time_step, susceptible_count);
        atomicAdd(output.total_exposed + time_step,     exposed_count);
        atomicAdd(output.total_infectious + time_step,  infectious_count);
        atomicAdd(output.total_recovered + time_step,   recovered_count);
    }
}

// Returns an integer in the range [min, max). Note - no parameter checking. :)
__device__ uint32_t randrange(curandState* prngState, uint32_t min, uint32_t max) {
    uint32_t draw = curand(prngState);
    draw %= (max - min);
    draw += min;

    return draw;
}
