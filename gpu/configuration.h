#pragma once

#include <string>
#include <vector>

#ifndef RAPIDJSON_HAS_STDSTRING
#define RAPIDJSON_HAS_STDSTRING 1
#endif
#ifndef RAPIDJSON_HAS_CXX11_RANGE_FOR
#define RAPIDJSON_HAS_CXX11_RANGE_FOR   1
#endif
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

    static float    Base_Infectivity;
    static uint32_t Default_Geography_Initial_Node_Population;
    static uint32_t Default_Geography_Torus_Size;
    static std::vector<std::string> Demographics_Filenames;
    static uint32_t Incubation_Period_Max;
    static uint32_t Incubation_Period_Min;
    static uint32_t Infectious_Period_Max;
    static uint32_t Infectious_Period_Min;
    static std::string Regional_Migration_Filename;
    static uint32_t Run_Number;
    static uint32_t Simulation_Duration;
    static float    x_Base_Population;

protected:

private:
    rapidjson::Document document;
};