#include "configuration.h"

#include <iostream>
#include <fstream>
#include <sstream>

#define PARAMETERS  "parameters"

uint32_t Configuration::Default_Geography_Initial_Node_Population = 1000;
uint32_t Configuration::Simulation_Duration                       =   90;
uint32_t Configuration::Default_Geography_Torus_Size              =   10;
std::vector<std::string> Configuration::Demographics_Filenames;
std::string Configuration::Regional_Migration_Filename;
uint32_t Configuration::Run_Number                                =    1;


Configuration::Configuration()
    : document()
{

}

Configuration::~Configuration()
{

}

#define GetUintParam( var, _default ) \
    if ( parameters.HasMember( #var ) ) \
        var = parameters[ #var ].GetUint(); \
    else \
        var = _default;

#define GetStringParam( var, _default ) \
    if ( parameters.HasMember( #var ) ) \
        var = parameters[ #var ].GetString(); \
    else \
        var = _default;

#define GetStringVectorParam( var, _default ) \
    if ( parameters.HasMember( #var ) ) \
        ReadStringVector( parameters, #var, var ); \
    else \
        var = _default;

void ReadStringVector( rapidjson::Value& json, const std::string& key, std::vector<std::string>& var )
{
    if ( !json.HasMember( key ) ) {
        std::cerr << "Did not find key '" << key.c_str() << "' in JSON." << std::endl;
        exit(-1);
    }

    auto stringVector = json[ key ].GetArray();

    for ( auto& entry : stringVector )
    {
        var.push_back( entry.GetString() );
    }
}

void Configuration::FromString( const std::string& input )
{
    document.Parse( input );

    if ( HasMember( PARAMETERS ) )
    {
        auto& parameters = (*this)[ PARAMETERS ];
        GetUintParam( Default_Geography_Initial_Node_Population, 1000 );
        GetUintParam( Simulation_Duration, 90 );
        GetUintParam( Default_Geography_Torus_Size, 10 );
        GetStringVectorParam( Demographics_Filenames, std::vector<std::string>() );
        GetStringParam( Regional_Migration_Filename, "" );
        GetUintParam( Run_Number, 1 );
    }
}

void Configuration::FromFile( const std::string& filename )
{
    // Happy path:
    std::ifstream file( filename );
    std::stringstream buffer;
    buffer << file.rdbuf();

    this->FromString( buffer.str() );
}

bool Configuration::HasMember( const std::string& key )
{
    return document.HasMember( key );
}

rapidjson::Value& Configuration::operator []( const std::string& key )
{
    return document[ key ];
}
