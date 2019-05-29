/*
** copyright banner
*/

#include <iostream>

#include "cxxopts.hpp"

#include "sim-config.h"
#include "configuration.h"

extern bool debug;

SimConfig::SimConfig()
    : sim_duration( 180 )
    , config_file_name()
    , population_file_name()
    , prevalence_file_name()
    , network_file_name()
    , population_scale( 1.0f )
{
    // this area intentionally left blank
}

void SimConfig::Initialize( int argc, const char* argv[] )
{
    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("C,config",     "Configuration file   ", cxxopts::value<std::string>()->default_value(""))
        ("d,debug",      "Turn on debug output")
        ("h,help",       "Show help.");

    auto args = options.parse(argc, argv);

    if ( args.count("help") )
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    config_file_name = args["config"].as<std::string>();
    if ( config_file_name.length() > 0 )
    {
        Configuration config;
        config.FromFile( config_file_name );

        sim_duration = Configuration::Simulation_Duration;
        prng_seed    = Configuration::Run_Number;

        if ( Configuration::Demographics_Filenames.size() > 0 )
        {
            population_file_name = Configuration::Demographics_Filenames[0];
            prevalence_file_name = Configuration::Demographics_Filenames[0];
        }

        if ( Configuration::Regional_Migration_Filename.length() > 0 )
        {
            network_file_name = Configuration::Regional_Migration_Filename;
        }

        population_scale = Configuration::x_Base_Population;

        incubation_duration_minimum = Configuration::Incubation_Period_Min;
        incubation_duration_maximum = Configuration::Incubation_Period_Max;
        infection_duration_minimum  = Configuration::Infectious_Period_Min;
        infection_duration_maximum  = Configuration::Infectious_Period_Max;
        infectivity = Configuration::Base_Infectivity;

        migration_scale = Configuration::x_Regional_Migration;
    }

    debug = bool( args.count("debug") );
}