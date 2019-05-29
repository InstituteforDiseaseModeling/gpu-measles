/*
** copyright banner
*/

#include <fstream>

#include "iarchive.h"
#include "config-json.h"

#include "json-reader.h"

ConfigJson::ConfigJson()
    : metadata()
    , parameters()
{
    // nothing here yet
}

ConfigJson::ConfigJson( const std::string& filename )
    : metadata()
    , parameters()
{
    FromFile( filename );
}

void ConfigJson::FromFile( const std::string& filename )
{
    std::ifstream file( filename );
    std::stringstream buffer;
    buffer << file.rdbuf();

    FromString( buffer.str() );
}

void ConfigJson::FromString( const std::string& text )
{
    IArchive* reader = new JsonReader( text.c_str() );
    ConfigJson::serialize( *reader, this );
}

IMPLEMENT_SERIALIZABLE(ConfigJson)
{
    ConfigJson& instance = *obj;
    ar.labelElement("metadata");   ConfigMetadata::serialize(   ar, &instance.metadata );
    ar.labelElement("parameters"); ConfigParameters::serialize( ar, &instance.parameters );
}

/******************************************************************************/

ConfigMetadata::ConfigMetadata()
    : author("unknown")
    , tool("unknown")
    , date("2000-01-01 00:00:00")
{
    // nothing here yet
}

IMPLEMENT_SERIALIZABLE(ConfigMetadata)
{
    ConfigMetadata& instance = (*obj);
    ar.startObject();
    ar.labelElement("author") & instance.author;
    ar.labelElement("tool")   & instance.tool;
    ar.labelElement("date")   & instance.date;
    ar.endObject();
}

/******************************************************************************/

ConfigParameters::ConfigParameters()
    : base_Infectivity( 1.0f )
    , demographics_Filenames()
    , enable_Immunity( false )
    , enable_Initial_Susceptibility_Distribution( false )
    , incubation_Period_Max( 7 )
    , incubation_Period_Min( 5 )
    , infectious_Period_Max( 15 )
    , infectious_Period_Min( 5 )
    , regional_Migration_Filename()
    , run_Number( 19411207 )
    , simulation_Duration( 365 )
    , x_Base_Population( 1.0f )
    , x_Regional_Migration( 1.0f )
{
    // nothing here yet
}

IMPLEMENT_SERIALIZABLE(ConfigParameters)
{
    ConfigParameters& instance = *obj;

    ar.startObject();
    ar.labelElement( "Base_Infectivity" )                           & instance.base_Infectivity;
    ar.labelElement( "Demographics_Filenames" )                     & instance.demographics_Filenames;
    ar.labelElement( "Enable_Immunity" )                            & instance.enable_Immunity;
    ar.labelElement( "Enable_Initial_Susceptibility_Distribution" ) & instance.enable_Initial_Susceptibility_Distribution;
    ar.labelElement( "Incubation_Period_Max" )                      & instance.incubation_Period_Max;
    ar.labelElement( "Incubation_Period_Min" )                      & instance.incubation_Period_Min;
    ar.labelElement( "Infectious_Period_Max" )                      & instance.infectious_Period_Max;
    ar.labelElement( "Infectious_Period_Min" )                      & instance.infectious_Period_Min;
    ar.labelElement( "Regional_Migration_Filename" )                & instance.regional_Migration_Filename;
    ar.labelElement( "Run_Number" )                                 & instance.run_Number;
    ar.labelElement( "Simulation_Duration" )                        & instance.simulation_Duration;
    ar.labelElement( "x_Base_Population" )                          & instance.x_Base_Population;
    ar.labelElement( "x_Regional_Migration" )                       & instance.x_Regional_Migration;
    ar.endObject();
}
