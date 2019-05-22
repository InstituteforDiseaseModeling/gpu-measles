/*
** copyright banner
*/

#pragma once

#include <string>

struct SimConfig {

    uint32_t sim_duration;
    uint32_t prng_seed;
    std::string config_file_name;
    std::string population_file_name;
    std::string prevalence_file_name;
    std::string network_file_name;
    float population_scale;
    uint32_t incubation_duration_minimum;
    uint32_t incubation_duration_maximum;
    uint32_t infection_duration_minimum;
    uint32_t infection_duration_maximum;
    float infectivity;
    float migration_scale;

    SimConfig();
    void Initialize( int argc, const char* argv[] );
};
