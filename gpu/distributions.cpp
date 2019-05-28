#include <chrono>
#include <iostream>

#include "distributions.h"

IDistribution* DistributionFactory::Create( DISTRIBUTION_TYPE distribution, float param1, float param2 )
{
    IDistribution* pDistribution = nullptr;

    switch (distribution)
    {
    case DISTRIBUTION_TYPE::DISTRIBUTION_CONSTANT:
        pDistribution = new DistributionConstant( param1 );
        break;

    case DISTRIBUTION_TYPE::DISTRIBUTION_UNIFORM:
        pDistribution = new DistributionUniform( param1, param2 );
        break;

    case DISTRIBUTION_TYPE::DISTRIBUTION_GAUSSIAN:
        pDistribution = new DistributionGaussian( param1, param2 );
        break;

    case DISTRIBUTION_TYPE::DISTRIBUTION_EXPONENTIAL:
        pDistribution = new DistributionExponential( param1 );
        break;

    case DISTRIBUTION_TYPE::DISTRIBUTION_POISSON:
        pDistribution = new DistributionPoisson( param1 );
        break;

    default:
        std::cerr << "Unsupported distribution (" << uint32_t(distribution) << ")." << std::endl;
        exit(-1);
    }

    return pDistribution;
}

DistributionBase::DistributionBase( void )
    : m_prng( std::chrono::system_clock::now().time_since_epoch().count() )
{
}

DistributionConstant::DistributionConstant( float constant )
    : m_constant( constant )
{
}

float DistributionConstant::draw( void )
{
    return m_constant;
}

DistributionUniform::DistributionUniform( float param1, float param2 )
    : DistributionBase()
    , m_min( std::min(param1, param2) )
    , m_max( std::max(param1, param2) )
    , distribution( m_min, m_max )
{
}

float DistributionUniform::draw( void )
{
    float value = distribution( m_prng );
    return value;
}

DistributionGaussian::DistributionGaussian( float param1, float param2 )
    : DistributionBase()
    , m_mean( param1 )
    , m_stddev( param2 )
    , distribution( m_mean, m_stddev )
{
}

float DistributionGaussian::draw( void )
{
    float value = distribution( m_prng );
    return value;
}

DistributionExponential::DistributionExponential( float param1 )
    : DistributionBase()
    , m_lambda( param1 )
    , distribution( m_lambda )
{
}

float DistributionExponential::draw( void )
{
    float value = distribution( m_prng );
    return value;
}

DistributionPoisson::DistributionPoisson( float param1 )
    : DistributionBase()
    , m_mean( param1 )
    , distribution( m_mean )
{
}

float DistributionPoisson::draw( void )
{
    float value = float(distribution( m_prng ));
    return value;
}