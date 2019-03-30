#pragma once

#include <cstdint>
#include <string>

struct Transmission
{
public:
    Transmission() {}
    virtual ~Transmission() {}
    Transmission(const Transmission&) = delete;
    Transmission&operator=(const Transmission&) = delete;

    virtual void DoTransmission( float* pInput, float* pOutput, uint32_t* pPopulations ) = 0;
};

class SpatialTx : public Transmission
{
public:
    SpatialTx(uint32_t num_locations, float* connectivity);
    virtual ~SpatialTx();

    virtual void DoTransmission( float* pInput, float* pOutput, uint32_t* pPopulations ) override;

protected:

    uint32_t count;
    uint32_t* pTotal;
    float* pFactors;
    float* pOutflow;
    float* pInflow;

private:
};

struct DummyTx : public Transmission
{
    DummyTx() {}
    virtual ~DummyTx() {}
    virtual void DoTransmission( float* pInput, float* pOutput, uint32_t* pPopulations ) override {}
};

Transmission* TransmissionFromFile( const std::string& filename, uint32_t num_locations );
