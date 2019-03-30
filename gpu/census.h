/*
** copyright
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sim-config.h"

struct Census
{
    uint32_t num_locales;
    std::vector<uint32_t> population;
    std::vector<float> prevalence;

    Census();
    void Initialize( const SimConfig& configuration );

protected:

    void populationFromJsonFile( const std::string& filename );
    void prevalenceFromJsonFile( const std::string& filename );
};
