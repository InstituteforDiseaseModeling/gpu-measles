/*
** copyright banner
*/

#include <iostream>

#include "cuda.h"
#include "spatial-tx.h"
#include "spatial-tx-kernel.h"
#include "utilities.h"

#include <cuda_runtime.h>
#include "check-return.h"

#define CSV_IO_NO_THREAD
#include "csv.h"

// #define DEBUG_PRINT (true)

void loadCsvConnectivityFile( const std::string& filename, uint32_t num_locations, std::vector<float>& connectivity );
void loadBinaryConnectivityFile( const std::string& filename, uint32_t num_locations, std::vector<float>& connectivity );

Transmission* TransmissionFromFile( const std::string& filename, uint32_t num_locations, float factor )
{
    Transmission* tx = nullptr;

    if ( filename.length() > 0 )
    {
        std::vector<float> connectivity;
        auto fileType = util::DetermineFileType( filename );
        switch ( fileType )
        {
            case util::typeCsv:
                loadCsvConnectivityFile( filename, num_locations, connectivity );
                break;

            case util::typeJson:
                std::cerr << "JSON format for connectivity not yet supported." << std::endl;
                break;

            default:
                loadBinaryConnectivityFile( filename, num_locations, connectivity );
                break;
        }

        tx = new SpatialTx( num_locations, connectivity.data(), factor );
    }
    else
    {
        tx = new DummyTx();
    }

    return tx;
}

SpatialTx::SpatialTx( uint32_t num_locations, float* connectivity, float factor )
    : count( num_locations )
    , pTotal( nullptr )
    , pFactors( nullptr )
    , pOutflow( nullptr )
    , pInflow( nullptr )
    , factor( factor )
{
    // TODO - would performance increase if factor and contagion matrix stride
    // was a factor of 32?
    cuda::mallocManaged(&pTotal,                     sizeof(uint32_t));  // scalar
    cuda::mallocManaged(&pFactors,   count * count * sizeof(float));     // matrix
    cuda::mallocManaged(&pOutflow,           count * sizeof(float));     // vector
    cuda::mallocManaged(&pInflow,            count * sizeof(float));     // vector

#ifdef DEBUG_PRINT
{
    if ( count <= 32 )
    {
        std::cerr << "Connectivity matrix:" << std::endl;
        for ( size_t j = 0; j < count; ++j )
        {
            for ( size_t i = 0; i < count; ++i )
            {
                std::cerr << connectivity[j * count + i] << ' ';
            }
            std::cerr << std::endl;
        }
    }
}
#endif

    cuda::memcpy( pFactors, connectivity, count * count * sizeof(float), cudaMemcpyDefault );
}

SpatialTx::~SpatialTx()
{
    cuda::free( pInflow );   pInflow    = nullptr;
    cuda::free( pOutflow );  pOutflow   = nullptr;
    cuda::free( pFactors );  pFactors   = nullptr;
    cuda::free( pTotal );    pTotal     = nullptr;
}

void SpatialTx::DoTransmission( float* pInput, float* pOutput, uint32_t* pPopulations )
{
    // Reduce populations in pPopulations to total (1-D operation)
    dim3 b( THREADS_PER_BLOCK, 1, 1 );
    dim3 g( (count + b.x - 1) / b.x, 1, 1);
    total_population<<<g, b>>>( count, pPopulations, pTotal );

    // Don't need this since cudaMemset below does not depend on this calculation
    // cuda::deviceSynchronize();

#ifdef DEBUG_PRINT
{
    std::cerr << "Total population: " << *pTotal << std::endl;
    std::cerr << "Node: pop - contagion" << std::endl;
    for ( size_t i = 0; i < count; ++i )
    {
        std::cerr << i << ": " << pPopulations[i] << " - " << pInput[i] << std::endl;
    }
}
#endif

    cuda::memset( pOutflow, 0, count * sizeof(float) );
    cuda::memset( pInflow,  0, count * sizeof(float) );

    cuda::deviceSynchronize();

    // Use connectivity information, current populations, and total population to calculate transferred contagion (2-D operation)
    b = dim3( THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1 );
    g = dim3( (count + b.x - 1) / b.x, (count + b.y - 1) / b.y, 1 );
    calculate_contagion<<<g, b>>>(
        count,
        pFactors,
        pInput,
        pPopulations,
        pTotal,
        pOutflow,
        pInflow,
        factor
    );

    cuda::deviceSynchronize();

#ifdef DEBUG_PRINT
{
    std::cerr << "Node: influx - outflow" << std::endl;
    for ( size_t i = 0; i < count; ++i )
    {
        std::cerr << i << ": " << pInflow[i] << " - " << pOutflow[i] << std::endl;
    }
}
#endif

    // Update contagion (1-D operation)
    b = dim3( THREADS_PER_BLOCK, 1, 1 );
    g = dim3( (count + b.x - 1) / b.x, 1, 1);
    update_contagion<<<g, b>>>( count, pInflow, pOutflow, pOutput );

    cuda::deviceSynchronize();

#ifdef DEBUG_PRINT
{
    std::cerr << "Node: contagion" << std::endl;
    for ( size_t i = 0; i < count; ++i )
    {
        std::cerr << i << ": " << pOutput[i] << " (" << pInput[i] << ")" << std::endl;
    }
}
#endif
}


void loadCsvConnectivityFile( const std::string& filename, uint32_t num_locations, std::vector<float>& connectivity )
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

void loadBinaryConnectivityFile( const std::string& filename, uint32_t num_locations, std::vector<float>& connectivity )
{
    std::cerr << "Loading binary connectivity file '" << filename.c_str() << "'..." << std::endl;
    // CheckFileSize assumes uint32_t, but will work for float also.
    auto file_size = util::CheckFileSize( filename );
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
