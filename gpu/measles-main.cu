#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <climits>
#include <iomanip>
#include <numeric>

#include <sys/stat.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "check-return.h"

#include "cxxopts.hpp"

#include "measles-init.h"
#include "measles-kernel.h"
#include "vax-kernel.h"
#include "block-size.h"

std::vector<uint32_t> LoadPopulationFromFile( const std::string& filename, uint32_t& agent_count, uint32_t& num_locations );

void AllocatePrngState(uint32_t agent_count, curandState** pPrngState);
void AllocateStateMemory(uint32_t agent_count, STATE& state);
void AllocateOutputMemory(uint32_t time_steps, uint32_t locations, OUTPUT& output);
void InitializePrngStates(uint32_t agent_count, curandState* pPrngStates);
void InitializeAges(uint32_t agent_count, STATE& state, curandState* pPrngStates);

void PopulatePopulation(uint32_t agent_count, uint32_t num_locations, uint32_t* populations);
void InitializeContagion(uint32_t num_locations, uint32_t* populations, float* contagion, float initial_prevalence);

void InitializeLocations(uint32_t agent_count, uint32_t num_locations, STATE& state, uint32_t* populations, curandState* pPrngStates);
void IncrementCurrent(uint32_t locations, OUTPUT& current);

void DisplayAges(uint32_t agent_count, STATE& state);
void DisplayLocations(uint32_t agent_count, STATE& state);

void PrintElapsedTime(clock_t start, clock_t end, const char* msg = nullptr, std::ostream& stream = std::cerr);

void dumpPopulations(uint32_t num_locations, uint32_t* population);
void dumpContagion(uint32_t num_locations, float* contagion);
void dumpOutput(uint32_t time_steps, uint32_t num_locations, OUTPUT& output);

bool debug = false;

int main(int argc, const char* argv[])
{
    uint32_t agent_count = 1<<20;   // 2^20 = 1024*1024 = 1048576
    uint32_t time_steps = 90;
    float initial_prevalence = 0.01f;
    uint32_t num_locations = 1;
//    uint32_t location_stride = 1;

    std::string population_file;

    uint32_t vaccination_node = 0;
    int32_t vaccination_time  = -1;
    uint32_t vaccine_duration = 730;
    uint32_t minimum_age      = 0;
    uint32_t maximum_age      = 42000;   //Just over 115 years

    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("c,count",      "Number of agents     ", cxxopts::value<int>()->default_value("1048576"))
        ("s,steps",      "Number of time steps ", cxxopts::value<int>()->default_value("90"))
        ("l,locations",  "Number of geographies", cxxopts::value<int>()->default_value("1"))
        ("p,prevalence", "Initial prevalence   ", cxxopts::value<float>()->default_value("0.01"))
        ("population",   "population file",       cxxopts::value<std::string>())
        ("n,node",       "Vaccination node     ", cxxopts::value<int>()->default_value("0"))
        ("t,time",       "SIA time             ", cxxopts::value<int>()->default_value("-1"))
        ("e,efficacy",   "Vaccine efficacy     ", cxxopts::value<int>()->default_value("730"))
        ("minimum",      "Minimum age          ", cxxopts::value<int>()->default_value("0"))
        ("maximum",      "Maximum age          ", cxxopts::value<int>()->default_value("42000"))
        ("d,debug",      "Turn on debug output")
        ("h,help",       "Show help.");

    auto args = options.parse(argc, argv);

    if ( args.count("help") )
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    agent_count        = uint32_t(args["count"].as<int>());
    time_steps         = uint32_t(args["steps"].as<int>());
    num_locations      = uint32_t(args["locations"].as<int>());
    initial_prevalence = args["prevalence"].as<float>();

    population_file    = args["population"].as<std::string>();

    vaccination_node   = uint32_t(args["node"].as<int>());
    vaccination_time   = int32_t(args["time"].as<int>());
    vaccine_duration   = uint32_t(args["efficacy"].as<int>());
    minimum_age        = uint32_t(args["minimum"].as<int>());
    maximum_age        = uint32_t(args["maximum"].as<int>());

//    if ( num_locations > 1 )
//    {
//        // round up to next multiple of 32
//        location_stride = (num_locations + 31) & ~31;
//    }

    debug = bool( args.count("debug") );

    std::vector<uint32_t> population = LoadPopulationFromFile( population_file, agent_count, num_locations );

    printf("Running %d agents for %d time steps. Initial prevalence %f.\n",
           agent_count, time_steps, initial_prevalence);

    curandState* prngState;

    STATE state;
    OUTPUT output;
    OUTPUT initial;

    AllocatePrngState(agent_count, &prngState);
    AllocateStateMemory(agent_count, state);
    AllocateOutputMemory(time_steps, num_locations, output);
    AllocateOutputMemory(1, num_locations, initial);

    InitializePrngStates(agent_count, prngState);
    InitializeAges(agent_count, state, prngState);

    PopulatePopulation(agent_count, num_locations, initial.population);
    InitializeContagion(num_locations, initial.population, initial.contagion, initial_prevalence);

    InitializeLocations(agent_count, num_locations, state, initial.population, prngState);

    if ( debug )
    {
        dumpPopulations(num_locations, initial.population);
        dumpContagion(num_locations, initial.contagion);
        dumpOutput(4, num_locations, output);
    }

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((agent_count+blockDims.x-1)/blockDims.x, 1, 1);

    uint32_t* populations = initial.population;
    float* contagion      = initial.contagion;
    OUTPUT current = output;

    auto start = clock();
    for ( size_t step = 0; step < time_steps; ++step )
    {
        if ( (vaccination_time >= 0) && (step == vaccination_time) )
        {
            vaccinate_poisson<<<gridDims, blockDims>>>(
                agent_count,
                vaccination_node,
                minimum_age,
                maximum_age,
                vaccine_duration,
                state,
                prngState);
        }

        do_step<<<gridDims, blockDims>>>(
            agent_count,
            step,
            populations,
            contagion,
            state,
            current,
            prngState);

        CheckReturn(cudaDeviceSynchronize());
        CheckReturn(cudaGetLastError());

            // Capture this time step's contagion to use in the next time step
        contagion = current.contagion;
        IncrementCurrent(num_locations, current);

        if ( step == 0 )
        {
            DisplayAges(agent_count, state);
        }
    }

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "Simulation");

    std::cout << "TimeStep";
    for ( size_t location = 0; location < num_locations; ++location)
    {
        std::cout << ",S" << location + 1;
        std::cout << ",E" << location + 1;
        std::cout << ",I" << location + 1;
        std::cout << ",R" << location + 1;
        std::cout << ",P" << location + 1;
    }
    std::cout << std::endl;
    uint32_t s = int(log10(time_steps)) + 1;
    uint32_t w = int(log10(agent_count)) + 1;
    for ( size_t step = 0; step < time_steps; ++step )
    {
        std::cout << std::setw(s) << step;
        for ( size_t location = 0; location < num_locations; ++location)
        {
            std::cout << ',' << std::setw(w) << (output.susceptible + num_locations * step)[location]
                      << ',' << std::setw(w) << (output.exposed     + num_locations * step)[location]
                      << ',' << std::setw(w) << (output.infectious  + num_locations * step)[location]
                      << ',' << std::setw(w) << (output.recovered   + num_locations * step)[location]
                      << ',' << std::setw(w) << (output.incidence   + num_locations * step)[location];
        }
        std::cout << std::endl;
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

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    zero_state<<<gridDims, blockDims>>>(agent_count, state);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "AllocateStateMemory");
}

// Allocate and zero time steps entries for output.
void AllocateOutputMemory(uint32_t time_steps, uint32_t locations, OUTPUT& output)
{
    auto start = clock();

    CheckReturn(cudaMallocManaged(&output.population,    time_steps * locations * sizeof(output.population[0])));
    CheckReturn(cudaMallocManaged(&output.susceptible,   time_steps * locations * sizeof(output.susceptible[0])));
    CheckReturn(cudaMallocManaged(&output.exposed,       time_steps * locations * sizeof(output.exposed[0])));
    CheckReturn(cudaMallocManaged(&output.infectious,    time_steps * locations * sizeof(output.infectious[0])));
    CheckReturn(cudaMallocManaged(&output.recovered,     time_steps * locations * sizeof(output.recovered[0])));
    CheckReturn(cudaMallocManaged(&output.incidence,     time_steps * locations * sizeof(output.incidence[0])));
    CheckReturn(cudaMallocManaged(&output.contagion,     time_steps * locations * sizeof(output.contagion[0])));

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((time_steps + blockDims.x - 1) / blockDims.x, 1, 1);
    zero_output<<<gridDims, blockDims>>>(time_steps, locations, output);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    auto end = clock();
    PrintElapsedTime(start, end, "AllocateOutputMemory");
}

void InitializePrngStates(uint32_t agent_count, curandState* pPrngStates)
{
    auto start = clock();

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
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

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
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

void InitializeLocations(uint32_t agent_count, uint32_t num_locations, STATE& state, uint32_t *populations, curandState* pPrngStates)
{
    auto start = clock();

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    initialize_locations<<<gridDims, blockDims>>>(agent_count, num_locations, state, populations, pPrngStates);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

    DisplayLocations(agent_count, state);

    auto end = clock();
    PrintElapsedTime(start, end, "InitializeLocations");
}

void IncrementCurrent(uint32_t locations, OUTPUT& current)
{
    current.population  += locations;
    current.susceptible += locations;
    current.exposed     += locations;
    current.infectious  += locations;
    current.recovered   += locations;
    current.incidence   += locations;
    current.contagion   += locations;
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

void PopulatePopulation(uint32_t agent_count, uint32_t num_locations, uint32_t* populations)
{
    uint32_t remaining_population = agent_count;
    uint32_t remaining_locations  = num_locations;
    for (uint32_t i = 0; i < num_locations; ++i)
    {
        uint32_t local_pop = remaining_population / remaining_locations;
        populations[i] = local_pop;
        remaining_population -= local_pop;
        --remaining_locations;
    }

    std::cout << "Local population counts:" << std::endl;
    for (uint32_t i = 0; i < min(8, num_locations); ++i)
    {
        std::cout << i << ": " << populations[i] << std::endl;
    }
    std::cout << std::endl;
}

void InitializeContagion(uint32_t num_locations, uint32_t* populations, float* contagion, float prevalence)
{
    for (uint32_t i = 0; i < num_locations; ++i)
    {
        contagion[i] = populations[i] * prevalence;
    }

    std::cout << "Local contagion:" << std::endl;
    for (uint32_t i = 0; i < min(8, num_locations); ++i)
    {
        std::cout << i << ": " << contagion[i] << std::endl;
    }
    std::cout << std::endl;
}

void dumpPopulations(uint32_t num_locations, uint32_t* populations)
{
    std::cout << "Local populations:" << std::endl;
    for ( size_t i = 0; i < num_locations; ++i )
    {
        std::cout << populations[i] << ' ';
    }
    std::cout << std::endl;
}

void dumpContagion(uint32_t num_locations, float* contagion)
{
    std::cout << "Local contagion:" << std::endl;
    for ( size_t i = 0; i < num_locations; ++i )
    {
        std::cout << contagion[i] << ' ';
    }
    std::cout << std::endl;
}

void dumpOutput(uint32_t time_steps, uint32_t num_locations, OUTPUT& output)
{
    std::cout << "Output:" << std::endl;
    for ( size_t step = 0; step < time_steps; ++step )
    {
        std::cout << step << ": ";
        for ( size_t location = 0; location < num_locations; ++location)
        {
            std::cout << "[ " << (output.population  + step * num_locations)[location]
                      << ", " << (output.susceptible + step * num_locations)[location]
                      << ", " << (output.exposed     + step * num_locations)[location]
                      << ", " << (output.infectious  + step * num_locations)[location]
                      << ", " << (output.recovered   + step * num_locations)[location]
                      << ", " << (output.incidence   + step * num_locations)[location]
                      << ", " << (output.contagion   + step * num_locations)[location] << "] ";
        }
        std::cout << std::endl;
    }
}

enum PopulationFileType {
    typeBinary = 0,
    typeCsv    = 1,
    typeJson   = 2
};

PopulationFileType DetermineFileType( const std::string& filename )
{
    PopulationFileType type = typeBinary;   // default

    auto dot = filename.find_last_of(".");

    if ( dot > 0 )
    {
        std::string suffix = filename.substr( dot + 1 );
        std::transform( suffix.begin(), suffix.end(), suffix.begin(), ::tolower );

        if ( suffix == std::string("csv") )
        {
            type = typeCsv;
        }
        else if ( suffix == std::string("json") )
        {
            type = typeJson;
        }
        else
        {
            // do nothing, use default, typeBinary
        }
    }
    else
    {
        // do nothing, use default, typeBinary
    }

    return type;
}

size_t CheckFileSize( const std::string& filename )
{
    struct stat file_stats;

    stat( filename.c_str(), &file_stats );

    if ( file_stats.st_size > 0 )
    {
        if ( (file_stats.st_size % sizeof(uint32_t)) == 0 )
        {
            // all is good, nothing to do
        }
        else
        {
            std::cerr << "File '" << filename.c_str() << "' is not an even number of uint32 values." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "File '" << filename.c_str() << "' is empty." << std::endl;
        exit(-1);
    }

    return file_stats.st_size;
}

void LoadBinaryPopulationFile( const std::string& filename, uint32_t& agent_count, uint32_t& num_locations, std::vector<uint32_t>& populations )
{
    auto size = CheckFileSize( filename );
    FILE* file = fopen( filename.c_str(), "rb" );
    if ( file != nullptr )
    {
        num_locations = size / sizeof(uint32_t);
        populations.resize( num_locations );
        std::cerr << "Reading population data for " << num_locations << " nodes..." << std::endl;
        auto count = fread( populations.data(), sizeof(uint32_t), num_locations, file );
        std::cerr << "Read population data for " << count << " nodes." << std::endl;
        fclose( file );
        if ( count != num_locations )
        {
            std::cerr << "Error reading population data from '" << filename.c_str() << "'." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Couldn't open " << filename.c_str() << " for reading." << std::endl;
        exit(-1);
    }
}

std::vector<uint32_t> LoadPopulationFromFile( const std::string& filename, uint32_t& agent_count, uint32_t& num_locations )
{
    std::vector<uint32_t> populations;
    if ( filename.length() > 0 )
    {
        auto fileType = DetermineFileType( filename );
        switch ( fileType )
        {
            case typeCsv:
                std::cerr << "CSV format for population not yet supported." << std::endl;
                exit(-1);
                break;

            case typeJson:
                std::cerr << "JSON format for population not yet supported." << std::endl;
                exit(-1);
                break;

            default:
                LoadBinaryPopulationFile( filename, agent_count, num_locations, populations );
                break;
        }
        agent_count = std::accumulate( populations.begin(), populations.end(), 0 );
    }
    else
    {
        uint32_t remaining_population = agent_count;
        uint32_t remaining_locations  = num_locations;
        for (uint32_t i = 0; i < num_locations; ++i)
        {
            uint32_t local_pop = remaining_population / remaining_locations;
            populations.push_back( local_pop );
            remaining_population -= local_pop;
            --remaining_locations;
        }
    }

    return populations;
}