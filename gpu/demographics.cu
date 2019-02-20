#include "demographics.h"

#include <fstream>
#include <sstream>

Demographics::Demographics()
    : document()
{
}

Demographics::~Demographics()
{
}

// static
void Demographics::FromString( const std::string& input )
{
    document.Parse( input );
}

// static
void Demographics::FromFile( const std::string& filename )
{
    // Happy path:
    std::ifstream file( filename );
    std::stringstream buffer;
    buffer << file.rdbuf();

    this->FromString( buffer.str() );
}

bool Demographics::HasMember( const std::string& key )
{
    return document.HasMember( key );
}

rapidjson::Value& Demographics::operator []( const std::string& key )
{
    return document[ key ];
}