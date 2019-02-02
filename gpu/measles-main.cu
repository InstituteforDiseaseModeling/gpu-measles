#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <climits>
#include <iomanip>

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
void InitializePrngStates(curandState* pPrngStates, uint32_t agent_count);
void InitializeAges(uint32_t agent_count, curandState* pPrngStates, STATE& state);
void InitializeLocations(uint32_t agent_count, curandState* pPrngStates, STATE& state, uint16_t max_locations);

void DisplayAges(uint32_t agent_count, STATE& state);
void DisplayLocations(uint32_t agent_count, STATE& state);

int main(int argc, const char* argv[])
{
    uint32_t agent_count = 1<<20;   // 2^20 = 1024*1024 = 1048576
    uint32_t time_steps = 90;
    float initial_prevalence = 0.01f;
    uint16_t max_locations = 1;

    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("c,count",      "Number of agents    ", cxxopts::value<int>()->default_value("1048576"))
        ("t,time",       "Number of time steps", cxxopts::value<int>()->default_value("90"))
        ("l,locations",  "Number of geographic", cxxopts::value<int>()->default_value("1"))
        ("p,prevalence", "Initial prevalence  ", cxxopts::value<float>()->default_value("0.01"))
        ("h,help",       "Show help.");

    auto args = options.parse(argc, argv);

    if ( args.count("help") )
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    agent_count        = uint32_t(args["count"].as<int>());
    time_steps         = uint32_t(args["time"].as<int>());
    max_locations      = uint16_t(args["locations"].as<int>());
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

    InitializePrngStates(prngState, agent_count);
    InitializeAges(agent_count, prngState, state);
    InitializeLocations(agent_count, prngState, state, max_locations);

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count+blockDims.x-1)/blockDims.x, 1, 1);

    force_of_infection[0] = initial_prevalence * agent_count;
    auto start = clock();
    for ( size_t step = 0; step < time_steps; ++step )
    {
        do_step<<<gridDims, blockDims>>>(
            agent_count,
            step,
            prngState,
            force_of_infection,
            state,
            output);

        if ( step == 0 )
        {
            CheckReturn(cudaDeviceSynchronize());
            CheckReturn(cudaGetLastError());
            DisplayAges(agent_count, state);
        }
    }
    auto end = clock();

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    printf("TimeStep,Susceptible,Exposed,Infectious,Recovered\n");
    for ( size_t step = 0; step < time_steps; ++step ) {
        printf("%zd,%d,%d,%d,%d\n",
            step,
            output.total_susceptible[step],
            output.total_exposed[step],
            output.total_infectious[step],
            output.total_recovered[step]);
    }
    std::cout << (1000 * uint64_t(end - start) / double(CLOCKS_PER_SEC)) << " milliseconds for simulation." << std::endl;
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
    // Using sizeof(state.field[0]) lets us change the datatypes more easily...

    CheckReturn(cudaMallocManaged(&state.is_infected,       agent_count * sizeof(state.is_infected[0])));
    CheckReturn(cudaMallocManaged(&state.elapsed_time,      agent_count * sizeof(state.elapsed_time[0])));
    CheckReturn(cudaMallocManaged(&state.exposure_period,   agent_count * sizeof(state.exposure_period[0])));
    CheckReturn(cudaMallocManaged(&state.infectious_period, agent_count * sizeof(state.infectious_period[0])));
    CheckReturn(cudaMallocManaged(&state.recovery_period,   agent_count * sizeof(state.recovery_period[0])));
    CheckReturn(cudaMallocManaged(&state.age_in_days,       agent_count * sizeof(state.age_in_days[0])));
    CheckReturn(cudaMallocManaged(&state.location,          agent_count * sizeof(state.location[0])));
    CheckReturn(cudaMallocManaged(&state.spatial_contagion, agent_count * sizeof(state.spatial_contagion[0])));
    CheckReturn(cudaMallocManaged(&state.immunity,          agent_count * sizeof(state.immunity[0])));
    CheckReturn(cudaMallocManaged(&state.infectivity,       agent_count * sizeof(state.infectivity[0])));

    CheckReturn(cudaMemset(state.is_infected,          0, agent_count * sizeof(state.is_infected[0])));
    CheckReturn(cudaMemset(state.elapsed_time,         0, agent_count * sizeof(state.elapsed_time[0])));
    CheckReturn(cudaMemset(state.exposure_period,      0, agent_count * sizeof(state.exposure_period[0])));
    CheckReturn(cudaMemset(state.infectious_period,    0, agent_count * sizeof(state.infectious_period[0])));
    CheckReturn(cudaMemset(state.recovery_period,      0, agent_count * sizeof(state.recovery_period[0])));
    // age_in_days will be initialized elsewhere
    // location will be initialized elsewhere
    CheckReturn(cudaMemset(state.spatial_contagion, 0xFF, agent_count * sizeof(state.spatial_contagion[0])));
    CheckReturn(cudaMemset(state.immunity,             0, agent_count * sizeof(state.immunity[0])));
    CheckReturn(cudaMemset(state.infectivity,          0, agent_count * sizeof(state.infectivity[0])));
}

// Allocate and zero time steps entries for output.
void AllocateOutputMemory(OUTPUT& output, uint32_t time_steps)
{
    CheckReturn(cudaMallocManaged(&output.total_susceptible,   time_steps * sizeof(output.total_susceptible[0])));
    CheckReturn(cudaMallocManaged(&output.total_exposed,       time_steps * sizeof(output.total_exposed[0])));
    CheckReturn(cudaMallocManaged(&output.total_infectious,    time_steps * sizeof(output.total_infectious[0])));
    CheckReturn(cudaMallocManaged(&output.total_recovered,     time_steps * sizeof(output.total_recovered[0])));

    CheckReturn(cudaMemset(output.total_susceptible,  0, time_steps * sizeof(output.total_susceptible[0])));
    CheckReturn(cudaMemset(output.total_exposed,      0, time_steps * sizeof(output.total_exposed[0])));
    CheckReturn(cudaMemset(output.total_infectious,   0, time_steps * sizeof(output.total_infectious[0])));
    CheckReturn(cudaMemset(output.total_recovered,    0, time_steps * sizeof(output.total_recovered[0])));
}

void InitializePrngStates(curandState* pPrngStates, uint32_t agent_count)
{
    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count+blockDims.x-1)/blockDims.x, 1, 1);

    initialize_prng<<<gridDims, blockDims>>>(pPrngStates, agent_count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
}

void InitializeAges(uint32_t agent_count, curandState* pPrngStates, STATE& state)
{
    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    initialize_ages<<<gridDims, blockDims>>>(agent_count, pPrngStates, state);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    DisplayAges(agent_count, state);
}

void DisplayAges(uint32_t agent_count, STATE& state)
{
    std::cout << "Age (in days):" << std::endl;
    for (uint32_t i = 0; i < min(32, agent_count); ++i)
    {
        std::cout << std::setw(2) << i << ": " << state.age_in_days[i] << std::endl;
    }
    std::cout << std::endl;
}

void InitializeLocations(uint32_t agent_count, curandState* pPrngStates, STATE& state, uint16_t max_locations)
{
    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    initialize_locations<<<gridDims, blockDims>>>(agent_count, pPrngStates, state, max_locations);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    DisplayLocations(agent_count, state);
}

void DisplayLocations(uint32_t agent_count, STATE& state)
{
    std::cout << "Location (ID):" << std::endl;
    for (uint32_t i = 0; i < min(32, agent_count); ++i)
    {
        std::cout << std::setw(2) << i << ": " << state.location[i] << std::endl;
    }
    std::cout << std::endl;
}
