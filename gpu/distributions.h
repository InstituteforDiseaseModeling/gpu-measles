#pragma once

#include <cstdint>
#include <random>

enum class DISTRIBUTION_TYPE : uint32_t {
    DISTRIBUTION_CONSTANT       = 0,
    DISTRIBUTION_UNIFORM        = 1,
    DISTRIBUTION_GAUSSIAN       = 2,
    DISTRIBUTION_EXPONENTIAL    = 3,
    DISTRIBUTION_POISSON        = 4,
    DISTRIBUTION_LOGNORMAL      = 5,
    DISTRIBUTION_BIMODAL        = 6,
    DISTRIBUTION_WEIBULL        = 7
};

struct IDistribution
{
    virtual ~IDistribution() {}
    virtual float draw( void ) = 0;
};

class DistributionFactory
{
public:
    static IDistribution* Create( DISTRIBUTION_TYPE distribution, float param1, float param2 = 0.0f );
protected:
    DistributionFactory() {}
};

class DistributionBase : public IDistribution
{
public:
protected:
    DistributionBase();
    ~DistributionBase() {}
    std::mt19937_64 m_prng;

private:
};

class DistributionConstant : public IDistribution
{
public:
    DistributionConstant( float constant );
    virtual ~DistributionConstant() {}

    virtual float draw( void );

protected:
    float m_constant;

private:
};

class DistributionUniform : public DistributionBase
{
public:
    DistributionUniform( float minimum, float maximum );
    virtual ~DistributionUniform() {}

    virtual float draw( void );

protected:
    float m_min;
    float m_max;
    std::uniform_real_distribution<float> distribution;

private:
};

class DistributionGaussian : public DistributionBase
{
public:
    DistributionGaussian( float mean, float stddev );
    virtual ~DistributionGaussian() {}

    virtual float draw( void );

protected:
    float m_mean;
    float m_stddev;
    std::normal_distribution<float> distribution;

private:
};

class DistributionExponential : public DistributionBase
{
public:
    DistributionExponential( float lambda );
    virtual ~DistributionExponential() {}

    virtual float draw( void );

protected:
    float m_lambda;
    std::exponential_distribution<float> distribution;

private:
};

class DistributionPoisson : public DistributionBase
{
public:
    DistributionPoisson( float mean );
    virtual ~DistributionPoisson() {}

    virtual float draw( void );

protected:
    float m_mean;
    std::poisson_distribution<uint32_t> distribution;

private:
};
