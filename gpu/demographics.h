#pragma once

#define RAPIDJSON_HAS_STDSTRING 1
#include "rapidjson/document.h"

class Demographics
{
public:
    Demographics();
    ~Demographics();

    void FromString( const std::string& input );
    void FromFile( const std::string& filename );

    bool HasMember( const std::string& key );

    rapidjson::Value& operator []( const std::string& key );

protected:

private:
    rapidjson::Document document;
};
