/*
** copyright banner
*/

#pragma once

#include "iserializable.h"

struct ConfigMetadata : public ISerializable
{
    std::string author;
    std::string tool;
    std::string date;

    ConfigMetadata();

    DECLARE_SERIALIZABLE(ConfigMetadata);
};

struct ConfigParameters : public ISerializable
{
    float                       base_Infectivity;
    std::vector<std::string>    demographics_Filenames;
    bool                        enable_Immunity;
    bool                        enable_Initial_Susceptibility_Distribution;
    uint32_t                    incubation_Period_Max;
    uint32_t                    incubation_Period_Min;
    uint32_t                    infectious_Period_Max;
    uint32_t                    infectious_Period_Min;
    std::string                 regional_Migration_Filename;
    uint32_t                    run_Number;
    uint32_t                    simulation_Duration;
    float                       x_Base_Population;
    float                       x_Regional_Migration;

    ConfigParameters();

    DECLARE_SERIALIZABLE(ConfigParameters);
};

struct ConfigJson : public ISerializable
{
    ConfigMetadata      metadata;
    ConfigParameters    parameters;

    ConfigJson();
    ConfigJson( const std::string& filename );

    DECLARE_SERIALIZABLE(ConfigJson);

protected:
    void FromFile( const std::string& filename );
    void FromString( const std::string& json );
};
