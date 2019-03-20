#pragma once

#include <string>
#include <vector>

#define RAPIDJSON_HAS_STDSTRING 1
#define RAPIDJSON_HAS_CXX11_RANGE_FOR   1
#include "rapidjson/document.h"

class Configuration
{
public:
    Configuration();
    ~Configuration();

    void FromString( const std::string& input );
    void FromFile( const std::string& filename );

    bool HasMember( const std::string& key );

    rapidjson::Value& operator []( const std::string& key );

    static uint32_t Default_Geography_Initial_Node_Population;
    static uint32_t Simulation_Duration;
    static uint32_t Default_Geography_Torus_Size;
    static std::vector<std::string> Demographics_Filenames;
    static std::string Regional_Migration_Filename;
    static uint32_t Run_Number;

protected:

private:
    rapidjson::Document document;
};