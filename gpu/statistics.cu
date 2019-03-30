/*
** copyright
*/

#include <fstream>
#include <iomanip>
#include <iostream>

#include "cuda.h"
#include "statistics.h"

Statistics::Statistics()
    : num_locales( 0 )
    , population( nullptr )
    , susceptible( nullptr )
    , exposed( nullptr )
    , infectious( nullptr )
    , recovered( nullptr )
    , incidence( nullptr )
    , contagion( nullptr )
{
// this area intentionally left blank
}

Statistics::Statistics( const Statistics& source )
    : num_locales( source.num_locales )
    , population( source.population )
    , susceptible( source.susceptible )
    , exposed( source.exposed )
    , infectious( source.infectious )
    , recovered( source.recovered )
    , incidence( source.incidence )
    , contagion( source.contagion )
{
    // this area intentionally left blank
}

Statistics Statistics::operator +(size_t offset)
{
    Statistics result( *this );
    result.population  += (offset * num_locales);
    result.susceptible += (offset * num_locales);
    result.exposed     += (offset * num_locales);
    result.infectious  += (offset * num_locales);
    result.recovered   += (offset * num_locales);
    result.incidence   += (offset * num_locales);
    result.contagion   += (offset * num_locales);

    return result;
}

StatisticsImpl::StatisticsImpl( uint32_t time_steps, uint32_t num_locales, const std::vector<uint32_t>& initial_population, const std::vector<float>& initial_prevalence )
    : capacity( time_steps )
{
    this->num_locales = num_locales;

    cuda::mallocManaged( &population,  capacity * num_locales * sizeof(population[0]) );
    cuda::mallocManaged( &susceptible, capacity * num_locales * sizeof(susceptible[0]) );
    cuda::mallocManaged( &exposed,     capacity * num_locales * sizeof(exposed[0]) );
    cuda::mallocManaged( &infectious,  capacity * num_locales * sizeof(infectious[0]) );
    cuda::mallocManaged( &recovered,   capacity * num_locales * sizeof(recovered[0]) );
    cuda::mallocManaged( &incidence,   capacity * num_locales * sizeof(incidence[0]) );
    cuda::mallocManaged( &contagion,   capacity * num_locales * sizeof(contagion[0]) );

    cuda::memset( population,  0, capacity * num_locales * sizeof(population[0]) );
    cuda::memset( susceptible, 0, capacity * num_locales * sizeof(susceptible[0]) );
    cuda::memset( exposed,     0, capacity * num_locales * sizeof(exposed[0]) );
    cuda::memset( infectious,  0, capacity * num_locales * sizeof(infectious[0]) );
    cuda::memset( recovered,   0, capacity * num_locales * sizeof(recovered[0]) );
    cuda::memset( incidence,   0, capacity * num_locales * sizeof(incidence[0]) );
    cuda::memset( contagion,   0, capacity * num_locales * sizeof(contagion[0]) );

    if ( initial_population.size() > 0 )
    {
        if ( initial_population.size() != num_locales )
        {
            std::cerr << "Internal error, population size (" << initial_population.size() << ") does not equal number of spatial locations, " << num_locales << std::endl;
            exit( -1 );
        }

        if ( initial_prevalence.size() != num_locales )
        {
            std::cerr << "Internal error, prevalences size (" << initial_prevalence.size() << ") does not equal number of spatial locations, " << num_locales << std::endl;
            exit( -1 );
        }

        cuda::memcpy( population, initial_population.data(), num_locales * sizeof(population[0]), cudaMemcpyDefault );

        if ( initial_prevalence.size() > 0 )
        {
            for ( size_t i = 0; i < num_locales; ++i )
            {
                contagion[ i ] = initial_population[ i ] * initial_prevalence[ i ];
            }
        }
    }

    cuda::deviceSynchronize();
}

StatisticsImpl::~StatisticsImpl()
{
    cuda::free( population );
    cuda::free( susceptible );
    cuda::free( exposed );
    cuda::free( infectious );
    cuda::free( recovered );
    cuda::free( incidence );
    cuda::free( contagion );
}

void StatisticsImpl::WriteToFile( const std::string& filename )
{
    std::filebuf fb;
    fb.open ( filename, std::ios::out );
    std::ostream sout( &fb );

    sout << "TimeStep";
    for ( size_t location = 1; location <= num_locales; ++location)
    {
        sout << ",S" << location;
        sout << ",E" << location;
        sout << ",I" << location;
        sout << ",R" << location;
        sout << ",P" << location;
    }
    sout << std::endl;
    uint32_t s = int(log10(capacity)) + 1;
    uint32_t w = 7; // N million // int(log10(agent_count)) + 1;
    for ( size_t step = 0; step < capacity; ++step )
    {
        sout << std::setw(s) << step;
        for ( size_t location = 0; location < num_locales; ++location)
        {
            sout << ',' << std::setw(w) << (susceptible + num_locales * step)[location]
                 << ',' << std::setw(w) << (exposed     + num_locales * step)[location]
                 << ',' << std::setw(w) << (infectious  + num_locales * step)[location]
                 << ',' << std::setw(w) << (recovered   + num_locales * step)[location]
                 << ',' << std::setw(w) << (incidence   + num_locales * step)[location];
        }
        sout << std::endl;
    }

    fb.close();
}