/*
** copyright banner
*/

#include <iostream>

#include "importation.h"
#include "cuda.h"

// template pattern

void Importation::Apply( uint16_t time_step, Population& population, Statistics& statistics )
{
    if ( IsApplicable( time_step, population, statistics ) )
    {
        std::vector<uint16_t> locales = Locales( time_step, population, statistics );
        for (uint16_t locale : locales)
        {
            DepositContagion( locale, time_step, population, statistics );
        }
    }
}

ContagionIncrease::ContagionIncrease( uint16_t period, std::vector<uint16_t> locales, float quantity )
    : period( period )
    , locales( locales )
    , contagion( quantity )
{

}

bool ContagionIncrease::IsApplicable( uint16_t time_step, Population& population, Statistics& statistics )
{
    // bool isApplicable = ((time_step >= period) && ((time_step % period) == 0));
    bool isApplicable = ((time_step % period) == 0);
    return isApplicable;
}

std::vector<uint16_t> ContagionIncrease::Locales( uint16_t time_step, Population& population, Statistics& statistics )
{
    size_t index = ((time_step / period) % locales.size());
    std::vector<uint16_t> locale;
    locale.push_back( locales[index] );
    return locale;
}

void ContagionIncrease::DepositContagion( uint16_t locale, uint16_t time_step, Population& population, Statistics& statistics )
{
    cuda::deviceSynchronize();
    Statistics current = statistics[time_step + 1];
    current.contagion[locale] += contagion;
    std::cout << "ContagionIncrease added " << contagion << " units of contagion to locale " << locale << " at time_step " << time_step << '.' << std::endl;
}
