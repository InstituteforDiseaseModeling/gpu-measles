#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <climits>
#include <iomanip>
#include <numeric>

#include <sys/stat.h>

#include "cuda.h"
#include <curand_kernel.h>
#include "cub/cub/cub.cuh"

#include "cxxopts.hpp"
#define CSV_IO_NO_THREAD
#include "csv.h"
#include "configuration.h"
#include "demographics.h"
#include "distributions.h"

#include "measles-init.h"
#include "measles-kernel.h"
#include "spatial-tx.h"
#include "vital-dynamics.h"
#include "vax-kernel.h"
#include "block-size.h"

#include "rapidjson/pointer.h"

std::vector<uint32_t> LoadPopulationFromFile( const std::string& filename, uint32_t& agent_count, uint32_t& num_locations );
std::vector<float>    LoadPrevalenceFromFile( const std::string& filename, uint32_t num_locations, float initial_prevalence );
Transmission*         LoadConnectivityFromFile( const std::string& filename, uint32_t num_locations );

void AllocatePrngState(uint32_t agent_count, curandState** pPrngState);
void AllocateStateMemory(uint32_t agent_count, STATE& state);
void AllocateOutputMemory(uint32_t time_steps, uint32_t locations, OUTPUT& output);
void InitializePrngStates(uint32_t agent_count, curandState* pPrngStates);
void InitializeAges(uint32_t agent_count, STATE& state, curandState* pPrngStates);

void InitializePopulations(const std::vector<uint32_t>& cpu_populations, uint32_t* gpu_populations);
void InitializeContagion(const std::vector<float>& prevalences, const std::vector<uint32_t>& populations, float* contagion);

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
    uint32_t time_steps = 90;
    float initial_prevalence = 0.01f;
    uint32_t num_locations = 1;

    uint32_t vaccination_node = 0;
    int32_t vaccination_time  = -1;
    uint32_t vaccine_duration = 730;
    uint32_t minimum_age      = 0;
    uint32_t maximum_age      = 42000;   //Just over 115 years

    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("C,config",     "Configuration file   ", cxxopts::value<std::string>()->default_value(""))
        ("s,steps",      "Number of time steps ", cxxopts::value<int>()->default_value("90"))
        ("l,locations",  "Number of geographies", cxxopts::value<int>()->default_value("1"))
        ("p,prev",       "Initial prevalence   ", cxxopts::value<float>()->default_value("0.01"))
        ("population",   "Population file      ", cxxopts::value<std::string>()->default_value(""))
        ("prevalence",   "Prevalence file      ", cxxopts::value<std::string>()->default_value(""))
        ("spatial",      "Connectivity file    ", cxxopts::value<std::string>()->default_value(""))
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

    std::string population_file;
    std::string prevalence_file;
    std::string connectivity_file;

    std::string config_file( args["config"].as<std::string>() );
    if ( config_file.length() > 0 )
    {
        Configuration config;
        config.FromFile( config_file );
        time_steps = Configuration::Simulation_Duration;
        if ( Configuration::Demographics_Filenames.size() > 0 )
        {
            population_file = Configuration::Demographics_Filenames[0];
            prevalence_file = Configuration::Demographics_Filenames[0];
        }
        if ( Configuration::Regional_Migration_Filename.length() > 0 )
        {
            connectivity_file = Configuration::Regional_Migration_Filename;
        }
    }

    if ( args.count( "steps" ) ) time_steps = uint32_t(args["steps"].as<int>());
    num_locations      = uint32_t(args["locations"].as<int>());
    initial_prevalence = args["prev"].as<float>();

    if ( args.count( "population" ) ) population_file   = args["population"].as<std::string>();
    if ( args.count( "prevalence" ) ) prevalence_file   = args["prevalence"].as<std::string>();
    if ( args.count( "spatial"    ) ) connectivity_file = args["spatial"].as<std::string>();

    vaccination_node   = uint32_t(args["node"].as<int>());
    vaccination_time   = int32_t(args["time"].as<int>());
    vaccine_duration   = uint32_t(args["efficacy"].as<int>());
    minimum_age        = uint32_t(args["minimum"].as<int>());
    maximum_age        = uint32_t(args["maximum"].as<int>());

    debug = bool( args.count("debug") );

    std::vector<uint32_t> initial_populations = LoadPopulationFromFile( population_file, agent_count, num_locations );
    std::vector<float>    initial_prevalences = LoadPrevalenceFromFile( prevalence_file, num_locations, initial_prevalence );
    Transmission* spatialTransmission = LoadConnectivityFromFile( connectivity_file, num_locations );

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

    InitializePopulations(initial_populations, initial.population);
    InitializeContagion(initial_prevalences, initial_populations, initial.contagion);

    InitializeLocations(agent_count, num_locations, state, initial.population, prngState);

    if ( debug )
    {
        dumpPopulations(num_locations, initial.population);
        dumpContagion(num_locations, initial.contagion);
        dumpOutput(4, num_locations, output);
    }

    // crude birthrate: 40/1000/year = 0.04/individual/year  = 0.00010958904109589041/individual/day
    // crude mortality: 15/1000/year = 0.015/individual/year
    //                               = 0.985/individual/year survival probability
    //                               = 0.99995859363039054212/individual/day survival probability
    //                               = 0.00004140636960945788/individual/day mortality probability
    VitalDynamics* vitalDynamics = new VitalDynamics( 0.00011f, 0.0000414f, num_locations, prngState, state );

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((agent_count+blockDims.x-1)/blockDims.x, 1, 1);

    uint32_t* populations = initial.population;
    float* contagion      = initial.contagion;
    OUTPUT current = output;

    auto start = clock();
    for ( size_t step = 0; step < time_steps; ++step )
    {
/*
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

            cuda::deviceSynchronize();
        }
*/
        population.Update( step, populations, contagion, current );

        spatialTransmission->DoTransmission( current.contagion, current.contagion, populations );
        uint32_t new_individuals = vitalDynamics->Update( populations, 1.0f, current, agent_count );

        // Capture this time step's contagion to use in the next time step
        contagion = current.contagion;
        IncrementCurrent(num_locations, current);
    }

    auto end = clock();
    PrintElapsedTime(start, end, "Simulation");

    std::cout << "TimeStep";
    for ( size_t location = 1; location <= num_locations; ++location)
    {
        std::cout << ",S" << location;
        std::cout << ",E" << location;
        std::cout << ",I" << location;
        std::cout << ",R" << location;
        std::cout << ",P" << location;
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

// Allocate and zero time steps entries for output.
void AllocateOutputMemory(uint32_t time_steps, uint32_t locations, OUTPUT& output)
{
    auto start = clock();

    cuda::mallocManaged(&output.population,    time_steps * locations * sizeof(output.population[0]));
    cuda::mallocManaged(&output.susceptible,   time_steps * locations * sizeof(output.susceptible[0]));
    cuda::mallocManaged(&output.exposed,       time_steps * locations * sizeof(output.exposed[0]));
    cuda::mallocManaged(&output.infectious,    time_steps * locations * sizeof(output.infectious[0]));
    cuda::mallocManaged(&output.recovered,     time_steps * locations * sizeof(output.recovered[0]));
    cuda::mallocManaged(&output.incidence,     time_steps * locations * sizeof(output.incidence[0]));
    cuda::mallocManaged(&output.contagion,     time_steps * locations * sizeof(output.contagion[0]));

    cuda::memset(output.population,    0, time_steps * locations * sizeof(output.population[0]));
    cuda::memset(output.susceptible,   0, time_steps * locations * sizeof(output.susceptible[0]));
    cuda::memset(output.exposed,       0, time_steps * locations * sizeof(output.exposed[0]));
    cuda::memset(output.infectious,    0, time_steps * locations * sizeof(output.infectious[0]));
    cuda::memset(output.recovered,     0, time_steps * locations * sizeof(output.recovered[0]));
    cuda::memset(output.incidence,     0, time_steps * locations * sizeof(output.incidence[0]));
    cuda::memset(output.contagion,     0, time_steps * locations * sizeof(output.contagion[0]));

    cuda::deviceSynchronize();

    auto end = clock();
    PrintElapsedTime(start, end, "AllocateOutputMemory");
}

/*
void InitializeAges(uint32_t agent_count, STATE& state, curandState* pPrngStates)
{
    auto start = clock();

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((agent_count + blockDims.x - 1) / blockDims.x, 1, 1);
    initialize_ages<<<gridDims, blockDims>>>(agent_count, state, pPrngStates);
    cuda::deviceSynchronize();

    DisplayAges(agent_count, state);

    auto end = clock();
    PrintElapsedTime(start, end, "InitializeAges");
}
*/

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

    initialize_locations<<<gridDims, blockDims>>>( agent_count, num_locations, populations, state );
    cuda::deviceSynchronize();

    uint32_t* keys_in  = nullptr;
    uint32_t* keys_out = nullptr;
    uint16_t* tmp_locations = nullptr;
    cuda::mallocManaged( &keys_in,       agent_count * sizeof(keys_in[0]) );
    cuda::mallocManaged( &keys_out,      agent_count * sizeof(keys_out[0]) );
    cuda::mallocManaged( &tmp_locations, agent_count * sizeof(tmp_locations[0]) );
    generate_indices<<<gridDims, blockDims>>>( agent_count, keys_in, pPrngStates );
    cuda::memcpy( tmp_locations, state.location, agent_count * sizeof(tmp_locations[0]), cudaMemcpyDeviceToDevice );

    void* scratch = nullptr;
    size_t scratch_bytes;
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, agent_count );
    cuda::malloc( &scratch, scratch_bytes );
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, keys_in, keys_out, tmp_locations, state.location, agent_count );
    cuda::deviceSynchronize();
    cuda::free( scratch );

    cuda::free( tmp_locations );
    cuda::free( keys_out );
    cuda::free( keys_in );

    auto end = clock();
    PrintElapsedTime(start, end, "InitializeLocations");

    DisplayLocations(agent_count, state);
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

void InitializePopulations(const std::vector<uint32_t>& cpu_populations, uint32_t* gpu_populations)
{
    cuda::memcpy(gpu_populations, cpu_populations.data(), cpu_populations.size() * sizeof(uint32_t), cudaMemcpyDefault);

    if ( debug )
    {
        std::cout << "Local population counts:" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(8, cpu_populations.size()); ++i)
        {
            std::cout << i << ": " << gpu_populations[i] << std::endl;
        }
        std::cout << std::endl;
    }
}

void InitializeContagion(const std::vector<float>& prevalences, const std::vector<uint32_t>& populations, float* contagion)
{
    for (uint32_t i = 0; i < populations.size(); ++i)
    {
        contagion[i] = populations[i] * prevalences[i];
    }

    if ( debug )
    {
        std::cout << "Local contagion:" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(8, populations.size()); ++i)
        {
            std::cout << i << ": " << contagion[i] << std::endl;
        }
        std::cout << std::endl;
    }
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

void LoadBinaryPrevalenceFile( const std::string& filename, std::vector<float>& prevalences )
{
    std::cerr << "Loading binary prevalence file '" << filename.c_str() << "'..." << std::endl;
    auto file_size = CheckFileSize( filename );
    FILE* file = fopen( filename.c_str(), "rb" );
    if ( file != nullptr )
    {
        size_t num_locations = file_size / sizeof(float);
        prevalences.resize( num_locations );
        std::cerr << "Reading prevalence data for " << num_locations << " nodes..." << std::endl;
        auto count = fread( prevalences.data(), sizeof(float), num_locations, file );
        std::cerr << "Read prevalence data for " << count << " nodes." << std::endl;
        fclose( file );
        if ( count != num_locations )
        {
            std::cerr << "Error reading prevalence data from '" << filename.c_str() << "'." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Couldn't open " << filename.c_str() << " for reading prevalence data." << std::endl;
        exit(-1);
    }
}

void LoadCsvPrevalenceFile( const std::string& filename, std::vector<float>& prevalences )
{
    std::cerr << "Loading CSV prevalence file '" << filename.c_str() << "'..." << std::endl;
    io::CSVReader<1> in(filename);
    in.read_header(io::ignore_extra_column, "prevalence");
    float prevalence;
    while( in.read_row(prevalence) )
    {
        prevalences.push_back( prevalence );
    }
}

bool KeyExists( const rapidjson::Value& object, const std::string& path )
{
    bool exists = (rapidjson::Pointer( path ).Get( object ) != nullptr);
    return exists;
}

const rapidjson::Value& GetChild( const rapidjson::Value& object, const std::string& path )
{
    const rapidjson::Value& child = (*rapidjson::Pointer( path ).Get( object ));
    return child;
}

IDistribution* DistributionForObject( const rapidjson::Value& object, const std::string& element, const std::string& path = "", IDistribution* defaultDistribution = nullptr )
{
    IDistribution* distribution = nullptr;
    std::string fullPath( path + "/" + element + "Flag" );
    if ( KeyExists( object, fullPath ) )
    {
        auto& child = GetChild( object, path );
        uint32_t type = child[ element + "Flag" ].GetUint();
        float param1 = 0.0f;
        float param2 = 0.0f;
        switch ( type )
        {
            case 1: // DISTRIBUTION_UNIFORM
            case 2: // DISTRIBUTION_GAUSSIAN
                // Needs both parameters
                if ( !child.HasMember( element + "2" ) )
                {
                    std::cerr << '"' << element.c_str() << "Flag\": " << distribution << " requires " << element.c_str() << "2 value." << std::endl;
                    exit(-1);
                }
                param2 = child[  element + "2" ].GetFloat();
                // drop through

            case 0: // DISTRIBUTION_CONSTANT
            case 3: // DISTRIBUTION_EXPONENTIAL
            case 4: // DISTRIBUTION_POISSON
                if ( !child.HasMember( element + "1" ) )
                {
                    std::cerr << '"' << element.c_str() << "Flag\": " << distribution << " requires " << element.c_str() << "1 value." << std::endl;
                    exit(-1);
                }
                param1 = child[ element + "1" ].GetFloat();
                distribution = DistributionFactory::Create( DISTRIBUTION_TYPE(type), param1, param2 );
                break;

            case 5: // DISTRIBUTION_LOGNORMAL
            case 6: // DISTRIBUTION_BIMODAL
            case 7: // DISTRIBUTION_WEIBULL
            default:
                std::cerr << "Unsupported distribution \"" << path.c_str() << '/' << element.c_str() << "Flag\": " << distribution << std::endl;
                exit(-1);
                break;
        }
    }

    if ( !distribution )
    {
        if ( defaultDistribution )
        {
            distribution = defaultDistribution; // Need some reference counting... :(
        }
        else
        {
            distribution = DistributionFactory::Create( DISTRIBUTION_TYPE::DISTRIBUTION_CONSTANT, 1.0f/128 );   // ~1%
        }
    }

    return distribution;
}

void LoadJsonPrevalenceFile( const std::string& filename, std::vector<float>& prevalences )
{
    Demographics demographics;
    demographics.FromFile( filename );

    IDistribution* defaultPrevalence = DistributionForObject( demographics, "PrevalenceDistribution", "/Defaults/IndividualAttributes" );

    if ( demographics.HasMember( "Nodes" ) )
    {
        auto& nodes = demographics[ "Nodes" ];
        if ( nodes.IsArray() )
        {
            for ( size_t i = 0; i < nodes.Size(); ++i )
            {
                auto& node = nodes[ i ];
                auto distribution = DistributionForObject( node, "PrevalenceDistribution", "/IndividualAttributes", defaultPrevalence );
                prevalences.push_back( distribution->draw() );
            }
        }
        else
        {
            std::cerr << "\"Nodes\" section in demographics files is not an array." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Demographics files does not contain a \"Nodes\" section." << std::endl;
        exit(-1);
    }
}

std::vector<float> LoadPrevalenceFromFile( const std::string& filename, uint32_t num_locations, float initial_prevalence )
{
    std::vector<float> prevalences;
    if ( filename.length() > 0 )
    {
        auto fileType = DetermineFileType( filename );
        switch ( fileType )
        {
            case typeCsv:
                LoadCsvPrevalenceFile( filename, prevalences );
                break;

            case typeJson:
                LoadJsonPrevalenceFile( filename, prevalences );
                break;

            default:
                LoadBinaryPrevalenceFile( filename, prevalences );
                break;
        }

        if ( prevalences.size() != num_locations )
        {
            std::cerr << "Number of entries in prevalence file, " << prevalences.size() << ", does not match number of spatial nodes (" << num_locations << ")." << std::endl;
            exit(-1);
        }

        auto in_range = [] (float f) -> bool { return ((f < 0.0f) || (f > 1.0f)); };
        if ( std::count_if( prevalences.begin(), prevalences.end(), in_range ) > 0 )
        {
            std::cerr << "Found out-of-range (0 <= prevalence <= 1) value(s) in prevalence file." << std::endl;
            exit(-1);
        }
    }
    else
    {
        for ( size_t location = 0; location < num_locations; ++location )
        {
            prevalences.push_back( initial_prevalence );
        }
    }

    return prevalences;
}

void LoadCsvConnectivityFile( const std::string& filename, uint32_t num_locations, std::vector<float>& connectivity )
{
    std::cerr << "Loading CSV connectivity file '" << filename.c_str() << "'..." << std::endl;
    io::LineReader in(filename);
    while( char* line = in.next_line() )
    {
        do
        {
            connectivity.push_back( float( atof( line ) ) );
            line = strchr( line, ',' );
        }
        while (line++); // Note this checks for line != nullptr but also (post-)increments line to skip the comma.
    }
    size_t num_elements = num_locations * num_locations;
    if ( connectivity.size() != num_elements )
    {
        std::cerr << "Connectivity file, '" << filename.c_str() << "', does not have the expected number of elements. "
                  << "Expected " << num_elements << " but read " << connectivity.size() << '.' << std::endl;
        exit(-1);
    }
}

void LoadBinaryConnectivityFile( const std::string& filename, uint32_t num_locations, std::vector<float>& connectivity )
{
    std::cerr << "Loading binary connectivity file '" << filename.c_str() << "'..." << std::endl;
    // CheckFileSize assumes uint32_t, but will work for float also.
    auto file_size = CheckFileSize( filename );
    FILE* file = fopen( filename.c_str(), "rb" );
    if ( file != nullptr )
    {
        size_t num_elements = file_size / sizeof(float);
        if ( num_elements == (num_locations * num_locations) )
        {
            connectivity.resize( num_elements );
            std::cerr << "Reading connectivity data for " << num_locations << " nodes(" << num_elements << " elements)..." << std::endl;
            auto count = fread( connectivity.data(), sizeof(float), num_elements, file );
            std::cerr << "Read connectivity data, " << count << " elements." << std::endl;
            fclose( file );
            if ( count != num_elements )
            {
                std::cerr << "Error reading connectivity data from '" << filename.c_str() << "'." << std::endl;
                exit(-1);
            }
        }
        else
        {
            std::cerr << "Connectivity file, '" << filename.c_str() << "' does not contain a square matrix." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Couldn't open " << filename.c_str() << " for reading connectivity data." << std::endl;
        exit(-1);
    }
}

Transmission* LoadConnectivityFromFile( const std::string& filename, uint32_t num_locations )
{
    Transmission* tx = nullptr;

    if ( filename.length() > 0 )
    {
        std::vector<float> connectivity;
        auto fileType = DetermineFileType( filename );
        switch ( fileType )
        {
            case typeCsv:
                LoadCsvConnectivityFile( filename, num_locations, connectivity );
                break;

            case typeJson:
                std::cerr << "JSON format for connectivity not yet supported." << std::endl;
                break;

            default:
                LoadBinaryConnectivityFile( filename, num_locations, connectivity );
                break;
        }

        tx = new SpatialTx( num_locations, connectivity.data() );
    }
    else
    {
        tx = new DummyTx();
    }

    return tx;
}
