/*
** copyright
*/

#include <stdio.h>
#include <iostream>

#include "cuda.h"
#include "tx-log.h"

TransmissionLog::TransmissionLog( void )
    : capacity( 0 )
    , pSize( nullptr )
    , pTimeStep( nullptr )
    , pLocale( nullptr )
    , pAgeAtInfection( nullptr )
{
    // this area intentionally left blank
}

TransmissionLog::TransmissionLog( const TransmissionLog& source )
    : capacity( source.capacity )
    , pSize( source.pSize )
    , pTimeStep( source.pTimeStep )
    , pLocale( source.pLocale )
    , pAgeAtInfection( source.pAgeAtInfection )
{
    // this area intentionally left blank
}

TransmissionLogImpl::TransmissionLogImpl( size_t size )
{
    cuda::mallocManaged( &pSize,                  sizeof(pSize[0]) );   // Just one of these
    cuda::mallocManaged( &pTimeStep,       size * sizeof(pTimeStep[0]) );
    cuda::mallocManaged( &pLocale,         size * sizeof(pLocale[0]) );
    cuda::mallocManaged( &pAgeAtInfection, size * sizeof(pAgeAtInfection[0]) );

    cuda::memset( pSize, 0, sizeof(pSize[0]) );

    capacity = uint32_t( size );
}

TransmissionLogImpl::~TransmissionLogImpl( void )
{
    cuda::free( pAgeAtInfection );
    cuda::free( pLocale );
    cuda::free( pTimeStep );
    cuda::free( pSize );
}

void TransmissionLogImpl::WriteToFile( const std::string& filename )
{
    FILE* file = fopen( filename.c_str(), "wb" );
    if ( file )
    {
        size_t size = size_t( *pSize );
        size_t count = fwrite( pTimeStep, sizeof(pTimeStep[0]), size, file );
        if ( count != size )
        {
            // TODO
        }
        count = fwrite( pLocale, sizeof(pLocale[0]), size, file );
        if ( count != size )
        {
            // TODO
        }
        count = fwrite( pAgeAtInfection, sizeof(pAgeAtInfection[0]), size, file );
        if ( count != size )
        {
            // TODO
        }
        fclose( file );
    }
    else
    {
        std::cerr << "Could not open '" << filename.c_str() << "' for writing." << std::endl;
    }
}