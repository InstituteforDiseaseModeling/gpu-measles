/*
** copyright banner
*/

#include <iostream>

#include "cxxopts.hpp"

#include "cmdline.h"

CmdLine::CmdLine( int argc, const char* argv[] )
    : config_filename()
    , debug( false )
{
    cxxopts::Options options(argv[0], "Agent based SEIR model on the GPU with CUDA.");
    options.add_options()
        ("C,config",     "Configuration file   ", cxxopts::value<std::string>()->default_value("config.json"))
        ("d,debug",      "Turn on debug output")
        ("h,help",       "Show help.");

    auto args = options.parse(argc, argv);

    if ( args.count("help") )
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    config_filename = args["config"].as<std::string>();
    debug = bool( args.count("debug") );
}