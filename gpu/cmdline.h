/*
** copyright banner
*/

#pragma once

#include <string>

struct CmdLine
{
    std::string config_filename;
    bool debug;

    CmdLine( int argc, const char* argv[] );
};
