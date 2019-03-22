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
#include "population.h"
#include "utilities.h"

#include "cub/cub/cub.cuh"

Population::Population( uint32_t capacity )
    : capacity( capacity )
    , size( 0 )
    , flags( nullptr )
    , elapsed_time( nullptr )
    , exposure_duration( nullptr )
    , infectious_duration( nullptr )
    , age_in_days( nullptr )
    , location( nullptr )
    , immunity( nullptr )
    , infectivity( nullptr )
    , pPrngStates( nullptr )
    , numLocales( 0 )
    , isView( false )
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

    cuda::deviceSynchronize();
}

Population::Population( const Population& source )
    : capacity( source.capacity )
    , size( source.size )
    , flags( source.flags )
    , elapsed_time( source.elapsed_time )
    , exposure_duration( source.exposure_duration )
    , infectious_duration( source.infectious_duration )
    , age_in_days( source.age_in_days )
    , location( source.location )
    , immunity( source.immunity )
    , infectivity( source.infectivity )
    , pPrngStates( source.pPrngStates )
    , numLocales( source.numLocales )
    , isView( true )
{

}

Population::~Population()
{
    if ( !isView )
    {
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
}

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
}


void Population::LoadFile( const std::string& filename )
{
    Census census;

    auto fileType = util::DetermineFileType( filename );
    switch ( fileType )
    {
//        case util::typeCsv:
//            loadCsvPopulationFile( filename, census );
//            break;

        case util::typeJson:
            loadJsonPopulationFile( filename, census );
            break;

        default:
//            loadBinaryPopulationFile( filename, census );
            std::cerr << "Unsupported file type for '" << filename.c_str() << "'." << std::endl;
            exit( -1 );
//            break;
    }
    num_locales = census.size();
    size = std::accumulate( census.begin(), census.end(), 0 );

    OUTPUT stats;
    instantiateNewIndividuals( 0, size, census.data(), stats );
}

void Population::loadJsonPopulationFile( const std::string& filename, std::vector<uint32_t>& populations )
{
    Demographics demographics;
    demographics.FromFile( filename );

    uint32_t defaultPopulation = 0;
    if ( demographics.HasMember( "Defaults" ) &&
         demographics[ "Defaults" ].HasMember( "NodeAttributes" ) &&
         demographics[ "Defaults" ][ "NodeAttributes" ].HasMember( "InitialPopulation" ) )
    {
        defaultPopulation = uint32_t(demographics[ "Defaults" ][ "NodeAttributes" ][ "InitialPopulation" ].GetInt64());
    }

    if ( demographics.HasMember( "Nodes" ) )
    {
        auto& nodes = demographics[ "Nodes" ];
        if ( nodes.IsArray() )
        {
            for ( size_t i = 0; i < nodes.Size(); ++i )
            {
                auto& node = nodes[ i ];
                if ( node.HasMember( "NodeAttributes" ) &&
                    node[ "NodeAttributes" ].HasMember( "InitialPopulation" ) )
                {
                    uint32_t population = uint32_t(node["NodeAttributes"][ "InitialPopulation" ].GetInt64());
                    if ( population > 0 )
                    {
                        populations.push_back( population );
                    }
                    else
                    {
                        std::cerr << "Initial population of node " << i << " <= 0." << std::endl;
                        exit(-1);
                    }
                }
                else
                {
                    std::cerr << "Using default population " << defaultPopulation << " for node " << i << "." << std::endl;
                    populations.push_back( defaultPopulation );
                }
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

__global__ void initialize_prng(uint32_t count, curandState* prngState)
{
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( tidx < count ) {
        curand_init( tidx, 0, 0, &prngState[ tidx ]);
    }
}

void Population::initializePrngStates()
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (capacity + blocks.x - 1) / blocks.x, 1, 1 );

    initialize_prng<<<grid, blocks>>>( capacity, pPrngStates );
    cuda::deviceSynchronize();
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

void Population::instantiateNewIndividuals( size_t offset, uint32_t count, int32_t* new_individuals, OUTPUT stats )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (size + blocks.x - 1) / blocks.x, 1, 1 );
    Population temp = *this + offset;
    new_individuals<<<grid, blocks>>>( size, num_locales, new_individuals, temp, pPrngStates );
    cuda::deviceSynchronize();

    scramblePopulation( offset );
    initializeNewIndividuals( 0, size, MEAN_AGE, false, stats );
}

void Population::scramblePopulation( size_t offset )
{
    void* scratch = nullptr;
    size_t scratch_bytes;
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, new_individuals );
    cuda::malloc( &scratch, scratch_bytes );
    Population temp = *this + offset;
    // Keys (random draws) are in immunity, use infectivity for output.
    // Locations are, initially, in age_in_days, store into "location".
    cub::DeviceRadixSort::SortPairs( scratch, scratch_bytes, temp.immunity, (uint16_t*)temp.infectivity, temp.age_in_days, temp.location, new_individuals );
    cuda::deviceSynchronize();
    cuda::free( scratch );
}

#define MAX_AGE     (100 * 365)

__global__ void initialize_newborns( uint32_t count, STATE state, uint16_t mean_age, bool initial_immunity, OUTPUT stats, curandState_t* pPrngStates )
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

void Population::initializeNewIndividuals( size_t offset, uint32_t count, uint16_t mean_age, bool initial_immunity, OUTPUT stats )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grid( (count + blocks.x - 1) / blocks.x, 1, 1 );
    Population temp = *this + offset;
    initialize_newborns<<<grid, blocks>>>( count, mean_age, initial_immunity, temp, stats, pPrngStates );
    cuda::deviceSynchronize();
}

void Population::Update( size_t time_step, uint32_t* populations, float* contagion, OUTPUT current )
{
    dim3 blocks( THREADS_PER_BLOCK, 1, 1 );
    dim3 grids( (size + blocks.x - 1) / blocks.x, 1, 1 );
    do_step<<<grid, blocks>>>(
        size,
        step,
        populations,
        contagion,
        *this,
        current,
        pPrngStates);

    cuda::deviceSynchronize();
}

/*
void LoadBinaryPopulationFile( const std::string& filename, std::vector<uint32_t>& populations )
{
    std::cerr << "Loading binary population file '" << filename.c_str() << "'..." << std::endl;
    auto file_size = CheckFileSize( filename );
    FILE* file = fopen( filename.c_str(), "rb" );
    if ( file != nullptr )
    {
        size_t num_locations = file_size / sizeof(uint32_t);
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
        std::cerr << "Couldn't open " << filename.c_str() << " for reading population data." << std::endl;
        exit(-1);
    }
}

void LoadCsvPopulationFile( const std::string& filename, std::vector<uint32_t>& populations )
{
    std::cerr << "Loading CSV population file '" << filename.c_str() << "'..." << std::endl;
    io::CSVReader<1> in(filename);
    in.read_header(io::ignore_extra_column, "population");
    uint32_t population;
    while( in.read_row(population) )
    {
        populations.push_back( population );
    }
}
*/