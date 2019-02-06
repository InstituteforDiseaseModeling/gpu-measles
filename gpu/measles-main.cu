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

void AllocatePrngState(uint32_t agent_count, curandState** pPrngState);
void AllocateForceOfInfection(uint32_t time_steps, float** pForce_of_Infection);
void AllocateStateMemory(uint32_t agent_count, STATE& state);
void AllocateOutputMemory(uint32_t time_steps, OUTPUT& output);
void InitializePrngStates(uint32_t agent_count, curandState* pPrngStates);
void InitializeAges(uint32_t agent_count, STATE& state, curandState* pPrngStates);
void InitializeLocations(uint32_t agent_count, uint16_t max_locations, STATE& state, curandState* pPrngStates);

void DisplayAges(uint32_t agent_count, STATE& state);
void DisplayLocations(uint32_t agent_count, STATE& state);

void PrintElapsedTime(clock_t start, clock_t end, const char* msg = nullptr, std::ostream& stream = std::cerr);

bool debug = false;

int main(int argc, const char* argv[])
{
    uint32_t agent_count = 1<<20;   // 2^20 = 1024*1024 = 1048576
    uint32_t time_steps = 90;
    float initial_prevalence = 0.01f;
    uint16_t max_locations = 1;

    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("c,count",      "Number of agents     ", cxxopts::value<int>()->default_value("1048576"))
        ("t,time",       "Number of time steps ", cxxopts::value<int>()->default_value("90"))
        ("l,locations",  "Number of geographies", cxxopts::value<int>()->default_value("1"))
        ("p,prevalence", "Initial prevalence   ", cxxopts::value<float>()->default_value("0.01"))
        ("d,debug",      "Turn on debug output")
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

    debug = bool( args.count("debug") );

    printf("Running %d agents for %d time steps. Initial prevalence %f.\n",
           agent_count, time_steps, initial_prevalence);

    curandState* prngState;

    float* force_of_infection;

    STATE state;
    OUTPUT output;

    AllocatePrngState(agent_count, &prngState);
    AllocateForceOfInfection(time_steps, &force_of_infection);
    AllocateStateMemory(agent_count, state);
    AllocateOutputMemory(time_steps, output);

    InitializePrngStates(agent_count, prngState);
    InitializeAges(agent_count, state, prngState);
    InitializeLocations(agent_count, max_locations, state, prngState);

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count+blockDims.x-1)/blockDims.x, 1, 1);

    force_of_infection[0] = initial_prevalence * agent_count;
    auto start = clock();
    for ( size_t step = 0; step < time_steps; ++step )
    {
        do_step<<<gridDims, blockDims>>>(
            agent_count,
            step,
            force_of_infection,
            state,
            output,
            prngState);

        if ( step == 0 )
        {
            CheckReturn(cudaDeviceSynchronize());
            CheckReturn(cudaGetLastError());
            DisplayAges(agent_count, state);
        }
    }

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "Simulation");

    printf("TimeStep,Susceptible,Exposed,Infectious,Recovered\n");
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
void AllocatePrngState(uint32_t agent_count, curandState** pPrngState)
{
    auto start = clock();

    CheckReturn(cudaMallocManaged(pPrngState, agent_count * sizeof(curandState)));
    // CheckReturn(cudaMemset(*pPrngState, 0, agent_count * sizeof(curandState)));

    auto end = clock();
    PrintElapsedTime(start, end, "AllocatePrngState");
}

// Allocate time_steps + 1 entries for force_of_infection (each time step
// updates the force of infection for the next time step).
void AllocateForceOfInfection(uint32_t time_steps, float** pForce_of_Infection)
{
    auto start = clock();

    CheckReturn(cudaMallocManaged(pForce_of_Infection, (time_steps + 1) * sizeof(float)));
    CheckReturn(cudaMemset(*pForce_of_Infection, 0, (time_steps + 1) * sizeof(float)));

    auto end = clock();
    PrintElapsedTime(start, end, "AllocateForceOfInfection");
}

// Allocate and zero enough state for each agent.
void AllocateStateMemory(uint32_t agent_count, STATE& state)
{
    // Using sizeof(state.field[0]) lets us change the datatypes more easily...

    auto start = clock();

    CheckReturn(cudaMallocManaged(&state.is_infected,         agent_count * sizeof(state.is_infected[0])));
    CheckReturn(cudaMallocManaged(&state.elapsed_time,        agent_count * sizeof(state.elapsed_time[0])));
    CheckReturn(cudaMallocManaged(&state.exposure_duration,   agent_count * sizeof(state.exposure_duration[0])));
    CheckReturn(cudaMallocManaged(&state.infectious_duration, agent_count * sizeof(state.infectious_duration[0])));
    CheckReturn(cudaMallocManaged(&state.age_in_days,         agent_count * sizeof(state.age_in_days[0])));
    CheckReturn(cudaMallocManaged(&state.location,            agent_count * sizeof(state.location[0])));
    CheckReturn(cudaMallocManaged(&state.immunity,            agent_count * sizeof(state.immunity[0])));
    CheckReturn(cudaMallocManaged(&state.infectivity,         agent_count * sizeof(state.infectivity[0])));

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    zero_state<<<gridDims, blockDims>>>(agent_count, state);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "AllocateStateMemory");
}

// Allocate and zero time steps entries for output.
void AllocateOutputMemory(uint32_t time_steps, OUTPUT& output)
{
    auto start = clock();

    CheckReturn(cudaMallocManaged(&output.total_susceptible,   time_steps * sizeof(output.total_susceptible[0])));
    CheckReturn(cudaMallocManaged(&output.total_exposed,       time_steps * sizeof(output.total_exposed[0])));
    CheckReturn(cudaMallocManaged(&output.total_infectious,    time_steps * sizeof(output.total_infectious[0])));
    CheckReturn(cudaMallocManaged(&output.total_recovered,     time_steps * sizeof(output.total_recovered[0])));

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((time_steps + blockDims.x - 1) / blockDims.x, 1, 1);
    zero_output<<<gridDims, blockDims>>>(time_steps, output);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "AllocateOutputMemory");
}

void InitializePrngStates(uint32_t agent_count, curandState* pPrngStates)
{
    auto start = clock();

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count+blockDims.x-1)/blockDims.x, 1, 1);

    initialize_prng<<<gridDims, blockDims>>>(agent_count, pPrngStates);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "InitializePrngStates");
}

void InitializeAges(uint32_t agent_count, STATE& state, curandState* pPrngStates)
{
    auto start = clock();

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    initialize_ages<<<gridDims, blockDims>>>(agent_count, state, pPrngStates);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    DisplayAges(agent_count, state);

    auto end = clock();
    PrintElapsedTime(start, end, "InitializeAges");
}

void DisplayAges(uint32_t agent_count, STATE& state)
{
    if ( !debug) return;

    std::cout << "Age (in days):" << std::endl;
    for (uint32_t i = 0; i < min(32, agent_count); ++i)
    {
        std::cout << std::setw(2) << i << ": " << state.age_in_days[i] << std::endl;
    }
    std::cout << std::endl;
}

void InitializeLocations(uint32_t agent_count, uint16_t max_locations, STATE& state, curandState* pPrngStates)
{
    auto start = clock();

    dim3 blockDims(BLOCK_SIZE, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    initialize_locations<<<gridDims, blockDims>>>(agent_count, max_locations, state, pPrngStates);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    DisplayLocations(agent_count, state);

    auto end = clock();
    PrintElapsedTime(start, end, "InitializeLocations");
}

void DisplayLocations(uint32_t agent_count, STATE& state)
{
    if ( !debug ) return;

    std::cout << "Location (ID):" << std::endl;
    for (uint32_t i = 0; i < min(32, agent_count); ++i)
    {
        std::cout << std::setw(2) << i << ": " << state.location[i] << std::endl;
    }
    std::cout << std::endl;
}

void PrintElapsedTime(clock_t start, clock_t end, const char* msg, std::ostream& stream)
{
    if ( msg )
    {
        stream << msg << " : ";
    }
    stream << (1000 * uint64_t(end - start) / double(CLOCKS_PER_SEC)) << " milliseconds" << std::endl;
}
