#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <climits>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "check-return.h"

#include "cxxopts.hpp"

#include "measles-kernel.h"
#include "block-size.h"

void AllocatePrngState(curandState** pPrngState, uint32_t agent_count);
void AllocateForceOfInfection(float** pForce_of_Infection, uint32_t time_steps);
void AllocateStateMemory(STATE& state, uint32_t agent_count);
void AllocateOutputMemory(OUTPUT& output, uint32_t time_steps);

int main(int argc, const char* argv[])
{
    uint32_t agent_count = 1024*1024;
    uint32_t time_steps = 90;
    float initial_prevalence = 0.01f;

    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("c,count",      "Number of agents     [2^10=1048576].", cxxopts::value<int>()->default_value("1048576"))
        ("t,time",       "Number of time steps [90].",           cxxopts::value<int>()->default_value("90"))
        ("p,prevalence", "Initial prevalence   [1%%].",          cxxopts::value<float>()->default_value("0.01"));

    auto args = options.parse(argc, argv);
    agent_count        = uint32_t(args["count"].as<int>());
    time_steps         = uint32_t(args["time"].as<int>());
    initial_prevalence = args["prevalence"].as<float>();

    printf("Running %d agents for %d time steps. Initial prevalence %f.\n",
           agent_count, time_steps, initial_prevalence);

    curandState* prngState;

    float* force_of_infection;

    STATE state;
    OUTPUT output;

    AllocatePrngState(&prngState, agent_count);
    AllocateForceOfInfection(&force_of_infection, time_steps);
    AllocateStateMemory(state, agent_count);
    AllocateOutputMemory(output, time_steps);

    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((agent_count+block.x-1)/block.x, 1, 1);

    initialize_prng<<<grid, block>>>(prngState, agent_count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    force_of_infection[0] = initial_prevalence * agent_count;
    for ( size_t step = 0; step < time_steps; ++step ) {
        do_step<<<grid, block>>>(
            agent_count,
            step,
            prngState,
            force_of_infection,
            state,
            output);
    }

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    printf("Time Step,Susceptible,Exposed,Infectious,Recovered\n");
    for ( size_t step = 0; step < time_steps; ++step ) {
        printf("%zd,%d,%d,%d,%d\n",
            step,
            output.total_susceptible[step],
            output.total_exposed[step],
            output.total_infectious[step],
            output.total_recovered[step]);
    }
}

// Allocate and zero one instance of curandState for each agent/thread
void AllocatePrngState(curandState** pPrngState, uint32_t agent_count)
{
    CheckReturn(cudaMallocManaged(pPrngState, agent_count * sizeof(curandState)));
    CheckReturn(cudaMemset(*pPrngState, 0, agent_count * sizeof(curandState)));
}

// Allocate time_steps + 1 entries for force_of_infection (each time step
// updates the force of infection for the next time step).
void AllocateForceOfInfection(float** pForce_of_Infection, uint32_t time_steps)
{
    CheckReturn(cudaMallocManaged(pForce_of_Infection, (time_steps + 1) * sizeof(float)));
    CheckReturn(cudaMemset(*pForce_of_Infection, 0, (time_steps + 1) * sizeof(float)));
}

// Allocate and zero enough state for each agent.
void AllocateStateMemory(STATE& state, uint32_t agent_count)
{
    CheckReturn(cudaMallocManaged(&state.is_infected,       agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&state.elapsed_time,      agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&state.exposure_period,   agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&state.infectious_period, agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&state.recovery_period,   agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&state.immunity,          agent_count * sizeof(float)));
    CheckReturn(cudaMallocManaged(&state.infectivity,       agent_count * sizeof(float)));

    CheckReturn(cudaMemset(state.is_infected,       0, agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMemset(state.elapsed_time,      0, agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMemset(state.exposure_period,   0, agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMemset(state.infectious_period, 0, agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMemset(state.recovery_period,   0, agent_count * sizeof(uint32_t)));
    CheckReturn(cudaMemset(state.immunity,          0, agent_count * sizeof(float)));
    CheckReturn(cudaMemset(state.infectivity,       0, agent_count * sizeof(float)));
}

// Allocate and zero time steps entries for output.
void AllocateOutputMemory(OUTPUT& output, uint32_t time_steps)
{
    CheckReturn(cudaMallocManaged(&output.total_susceptible,   time_steps * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&output.total_exposed,       time_steps * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&output.total_infectious,    time_steps * sizeof(uint32_t)));
    CheckReturn(cudaMallocManaged(&output.total_recovered,     time_steps * sizeof(uint32_t)));

    CheckReturn(cudaMemset(output.total_susceptible,  0, time_steps * sizeof(uint32_t)));
    CheckReturn(cudaMemset(output.total_exposed,      0, time_steps * sizeof(uint32_t)));
    CheckReturn(cudaMemset(output.total_infectious,   0, time_steps * sizeof(uint32_t)));
    CheckReturn(cudaMemset(output.total_recovered,    0, time_steps * sizeof(uint32_t)));
}