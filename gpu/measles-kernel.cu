#include "measles-kernel.h"
#include "cub/cub/cub.cuh"
#include "block-size.h"

__device__ inline void SusceptibleToInfected(uint32_t tidx, Population state, curandState* pPrngState);
__device__ inline float UpdateInfected(uint32_t tidx, Population state, bool& is_infected, bool& is_infectious);
__device__ inline uint32_t randrange(curandState* pPrngState, uint32_t min, uint32_t max);

__global__ void update_population(
    uint16_t timeStep,
    Population state,
    Statistics current,
    Statistics output,
    TransmissionLog transmission)
{
    int tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( tidx < state.size ) {  // Number of active agents

        uint32_t susceptible = 0;
        uint32_t exposed     = 0;
        uint32_t infectious  = 0;
        uint32_t recovered   = 0;

        float infectivity = 0.0f;

        uint32_t location = state.location[tidx];

        bool is_infected   = bool(state.flags[tidx] & INFECTED);
        bool is_immune     = false;
        bool is_infectious = false;
        bool innoculated   = false;

        // Everyone (even children) gets older...
        uint16_t age = ++state.age_in_days[tidx];

        if ( !is_infected )
        {
            is_immune = bool(state.immunity[tidx]);

            if ( !is_immune )
            {
                float draw = curand_uniform(&state.pPrngStates[tidx]);
                float susceptibility = current.contagion[location] / current.population[location];

                if ( draw <= susceptibility )
                {
                    SusceptibleToInfected(tidx, state, &state.pPrngStates[tidx]);
                    innoculated = true;
                    is_infected = true;

                    uint32_t slot = atomicAdd( transmission.pSize, 1 );
                    if ( slot < transmission.capacity )
                    {
                        transmission.pTimeStep[ slot ]       = timeStep;
                        transmission.pLocale[ slot ]         = location;
                        transmission.pAgeAtInfection[ slot ] = age;
                    }
                }
            }
            else
            {
                --state.immunity[tidx];
            }
        }
        else
        {
            is_immune = true;
            infectivity = UpdateInfected(tidx, state, is_infected, is_infectious);
        }

        susceptible = (!is_infected && !is_immune)     ? 1 : 0;
        exposed     = ( is_infected && !is_infectious) ? 1 : 0;
        infectious  = ( is_infected &&  is_infectious) ? 1 : 0;
        recovered   = (!is_infected &&  is_immune)     ? 1 : 0;

        if ( susceptible )       atomicAdd(output.susceptible + location, 1);
        if ( exposed )           atomicAdd(output.exposed     + location, 1);
        if ( infectious )        atomicAdd(output.infectious  + location, 1);
        if ( recovered )         atomicAdd(output.recovered   + location, 1);
        if ( innoculated )       atomicAdd(output.incidence   + location, 1);
        if ( infectivity > 0.0f) atomicAdd(output.contagion   + location, infectivity);
    }
}

__device__ inline void SusceptibleToInfected(uint32_t tidx, Population state, curandState* pPrngState)
{
    state.flags[tidx]               |= INFECTED;
    state.elapsed_time[tidx]         = 0;
    state.exposure_duration[tidx]    = randrange(pPrngState, 2, 5);
    state.infectious_duration[tidx]  = randrange(pPrngState, 5, 10);
    state.infectivity[tidx]          = 0.0f;
    state.immunity[tidx]             = UINT16_MAX;
}

__device__ inline float UpdateInfected(uint32_t tidx, Population state, bool& is_infected, bool& is_infectious)
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
            state.flags[tidx] &= ~INFECTED;
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

// Returns an integer in the range [min, max).
__device__ inline uint32_t randrange(curandState* pPrngState, uint32_t minimum, uint32_t maximum) {
    uint32_t draw = curand(pPrngState);
    uint32_t mn = min(minimum, maximum);
    uint32_t mx = max(minimum, maximum);
    draw %= (mx - mn);
    draw += mn;

    return draw;
}
