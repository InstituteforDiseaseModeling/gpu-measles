/***************************************************************************************************

Copyright (c) 2018 Intellectual Ventures Property Holdings, LLC (IVPH) All rights reserved.

EMOD is licensed under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License.
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

***************************************************************************************************/

#pragma once

#include <map>
#include <string>
#include <stack>        // for _pool declaration/implementation

#include "exceptions.h"

struct IArchive;

struct ISerializable
{
    virtual const char* GetClassName() { throw NotYetImplementedException( __FILE__, __LINE__, __FUNCTION__, "subclass needs to implement" ); }
    virtual void Recycle() { throw NotYetImplementedException( __FILE__, __LINE__, __FUNCTION__, "subclass needs to implement" ); }
    virtual ~ISerializable() {}

    static void serialize(IArchive&, ISerializable*&);
};

struct SerializationRegistrar
{
    typedef void (*serialize_function_t)(IArchive&, ISerializable*);
    typedef ISerializable* (*constructor_function_t)(void);

    static void _register(const char* classname, serialize_function_t serializer, constructor_function_t constructor)
    {
        if (!_singleton) _singleton = new SerializationRegistrar();
        _singleton->serializer_map[classname] = serializer;
        _singleton->constructor_map[classname] = constructor;
    }

    static serialize_function_t _get_serializer(std::string& classname) { return _singleton->serializer_map[classname]; }
    static constructor_function_t _get_constructor(std::string& classname) { return _singleton->constructor_map[classname]; }
    std::map<std::string, serialize_function_t> serializer_map;
    std::map<std::string, constructor_function_t> constructor_map;
    static SerializationRegistrar* _singleton;
};

template<typename Derived>
struct SerializationRegistrationCaller
{
    SerializationRegistrationCaller()
    {
        in_class_registration_hook();
    }
    static void in_class_registration_hook()
    {
        // typename breaks gcc -- need to make cross-platform
        SerializationRegistrar::_register(
            /*typename*/ Derived::_class_name,
            [](IArchive& ar, ISerializable* obj)
            {
                return Derived::serialize(ar, dynamic_cast<Derived*>(obj));
            },
            /*typename*/  Derived::construct); 
    }
};

#define DECLARE_SERIALIZABLE(classname)                                                         \
private:                                                                                        \
    virtual const char* GetClassName() override { return _class_name; }                         \
    static const char* _class_name;                                                             \
    friend SerializationRegistrationCaller<classname>;                                          \
    static SerializationRegistrationCaller<classname> serialization_registration_caller;        \
public:                                                                                         \
    static ISerializable* construct() { return dynamic_cast<ISerializable*>(new classname()); } \
    static void serialize(IArchive&, classname*)

#define IMPLEMENT_SERIALIZABLE(classname)               \
    const char* classname::_class_name = #classname;    \
    void classname::serialize(IArchive& ar, classname* obj)

#define REGISTER_SERIALIZABLE(classname)                                                 \
    SerializationRegistrar::_register( #classname, [](IArchive& ar, ISerializable* obj) { return classname::serialize(ar, dynamic_cast<classname*>(obj)); }, classname::construct )
