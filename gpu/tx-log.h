/*
** copyright
*/

#pragma once

#include <cstdint>
#include <string>

struct TransmissionLog
{
    uint32_t    capacity;
    uint32_t*   pSize;
    uint16_t*   pTimeStep;
    uint16_t*   pLocale;
    uint16_t*   pAgeAtInfection;

    TransmissionLog( void );
    TransmissionLog( const TransmissionLog& source );
    ~TransmissionLog( void ) {}
};

struct TransmissionLogImpl : public TransmissionLog
{
    TransmissionLogImpl( size_t size );
    ~TransmissionLogImpl( void );

    void WriteToFile( const std::string& filename );
};
