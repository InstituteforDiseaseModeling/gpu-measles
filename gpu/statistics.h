/*
** copyright
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Statistics
{
    uint32_t num_locales;
    uint32_t* population;
    uint32_t* susceptible;
    uint32_t* exposed;
    uint32_t* infectious;
    uint32_t* recovered;
    uint32_t* incidence;
    float*    contagion;

    Statistics();
    Statistics( const Statistics& source );

    Statistics operator +(size_t offset);
    Statistics operator [](size_t index);
};

struct StatisticsImpl : public Statistics
{
    StatisticsImpl( uint32_t time_steps, uint32_t num_locales, const std::vector<uint32_t>& initial_population = std::vector<uint32_t>(), const std::vector<float>& initial_prevalence = std::vector<float>() );
    ~StatisticsImpl();

    void WriteToFile( const std::string& filename );

protected:

    uint32_t capacity;
};
