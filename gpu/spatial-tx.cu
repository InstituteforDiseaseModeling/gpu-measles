#include <iostream>

#include "spatial-tx.h"
#include "spatial-tx-kernel.h"

#include <cuda_runtime.h>
#include "check-return.h"

SpatialTx::SpatialTx( uint32_t num_locations, float* connectivity )
    : count( num_locations )
    , pTotal( nullptr )
    , pFactors( nullptr )
    , pOutflow( nullptr )
    , pInflow( nullptr )
{
    // TODO - would performance increase if factor and contagion matrix stride
    // was a factor of 32?
    CheckReturn(cudaMallocManaged(&pTotal,                     sizeof(uint32_t)));  // scalar
    CheckReturn(cudaMallocManaged(&pFactors,   count * count * sizeof(float)));     // matrix
    CheckReturn(cudaMallocManaged(&pOutflow,           count * sizeof(float)));     // vector
    CheckReturn(cudaMallocManaged(&pInflow,            count * sizeof(float)));     // vector

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

    CheckReturn( cudaMemcpy( pFactors, connectivity, count * count * sizeof(float), cudaMemcpyDefault ) );
}

SpatialTx::~SpatialTx()
{
    cudaFree( pInflow );   pInflow    = nullptr;
    cudaFree( pOutflow );  pOutflow   = nullptr;
    cudaFree( pFactors );  pFactors   = nullptr;
    cudaFree( pTotal );    pTotal     = nullptr;
}

// #define DEBUG_PRINT (true)

void SpatialTx::DoTransmission( float* pInput, float* pOutput, uint32_t* pPopulations )
{
    // Reduce populations in pPopulations to total (1-D operation)
    dim3 b( THREADS_PER_BLOCK, 1, 1 );
    dim3 g( (count + b.x - 1) / b.x, 1, 1);
    total_population<<<g, b>>>( count, pPopulations, pTotal );

    // Don't need these since cudaMemset below does not depend on this calculation
    // CheckReturn(cudaDeviceSynchronize());
    // CheckReturn(cudaGetLastError());

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

    CheckReturn(cudaMemset( pOutflow, 0, count * sizeof(float) ));
    CheckReturn(cudaMemset( pInflow,  0, count * sizeof(float) ));

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

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
        pInflow
    );

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

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

    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());

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