/*
** copyright banner
*/

#pragma once

#include <cstdint>
#include <vector>

#include "population.h"
#include "statistics.h"

struct Importation
{
    virtual ~Importation() {}
    void Apply( uint16_t time_step, Population& population, Statistics& statistics );

protected:
    virtual bool IsApplicable( uint16_t time_step, Population& population, Statistics& statistics ) = 0;
    virtual std::vector<uint16_t> Locales( uint16_t time_step, Population& population, Statistics& statistics ) = 0;
    virtual void DepositContagion( uint16_t locale, uint16_t time_step, Population& population, Statistics& statistics ) = 0;
};

class ContagionIncrease : public Importation
{
public:
    ContagionIncrease( uint16_t period, std::vector<uint16_t> locales, float quantity );
    virtual ~ContagionIncrease() {}

protected:
    virtual bool IsApplicable( uint16_t time_step, Population& population, Statistics& statistics ) override;
    virtual std::vector<uint16_t> Locales( uint16_t time_step, Population& population, Statistics& statistics ) override;
    virtual void DepositContagion( uint16_t locale, uint16_t time_step, Population& population, Statistics& statistics ) override;

    uint16_t period;
    std::vector<uint16_t> locales;
    float contagion;
};

/*
// Instantiate new infected individual in chosen location
class Immigration : public Importation
{
protected:
    virtual bool IsApplicable( uint16_t time_step, Population& population, Statistics& statistics ) override;
    virtual std::vector<uint16_t> Locales( uint16_t time_step, Population& population, Statistics& statistics ) override;
    virtual void DepositContagion( uint16_t locale, uint16_t time_step, Population& population, Statistics& statistics ) override;
};

// Infect a stochastically selected susceptible individual in chosen location.
class Innoculation : public Importation
{
protected:
    virtual bool IsApplicable( uint16_t time_step, Population& population, Statistics& statistics ) override;
    virtual std::vector<uint16_t> Locales( uint16_t time_step, Population& population, Statistics& statistics ) override;
    virtual void DepositContagion( uint16_t locale, uint16_t time_step, Population& population, Statistics& statistics ) override;
};
*/
