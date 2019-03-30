#pragma once

#ifndef RAPIDJSON_HAS_STDSTRING
#define RAPIDJSON_HAS_STDSTRING 1
#endif
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
    operator rapidjson::Value&() { return document; }

protected:

private:
    rapidjson::Document document;
};
