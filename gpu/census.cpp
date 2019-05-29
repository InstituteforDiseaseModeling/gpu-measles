/*
** copyright
*/

#include <algorithm>

#ifndef RAPIDJSON_HAS_STDSTRING
#define RAPIDJSON_HAS_STDSTRING 1
#endif
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

#include "census.h"
#include "demographics.h"
#include "distributions.h"
#include "utilities.h"

Census::Census()
    : num_locales( 0 )
    , population()
    , prevalence()
{
    // this area intentionally left blank
}

#define DEFAULT_TORUS_SIZE  10
#define DEFAULT_NODE_POPULATION 1000
#define DEFAULT_NODE_PREVALENCE 0.0078125f  // 1/128 ~ 1%

void validatePrevalence( const std::vector<float>& prevalence, uint32_t num_locales );

void Census::Initialize( const ConfigParameters& parameters )
{
    float scale = parameters.x_Base_Population;

    const std::string& population_file_name = parameters.demographics_Filenames[0];

    if ( population_file_name.size() > 0 )
    {
        if ( parameters.regional_Migration_Filename.size() == 0 )
        {
            std::cerr << "Missing network specification file ('Regional_Migration_Filename' in parameters)." << std::endl;
            exit( -1 );
        }

        auto fileType = util::DetermineFileType( population_file_name );
        switch ( fileType )
        {
//        case util::typeCsv:
//            loadCsvPopulationFile( population_file_name, census );
//            break;
    
        case util::typeJson:
            populationFromJsonFile( population_file_name );
            prevalenceFromJsonFile( population_file_name );
            break;
    
        default:
//            loadBinaryPopulationFile( population_file_name, census );
            std::cerr << "Unsupported file type for '" << population_file_name.c_str() << "'." << std::endl;
            exit( -1 );
//            break;
        }

        for ( auto& entry : population )
        {
            entry *= scale;
        }

        num_locales = population.size();
        validatePrevalence( prevalence, num_locales );
    }
    else
    {
        num_locales = DEFAULT_TORUS_SIZE * DEFAULT_TORUS_SIZE;
        for ( size_t i = 0; i < num_locales; ++i )
        {
            population.push_back( DEFAULT_NODE_POPULATION * scale );
            prevalence.push_back( DEFAULT_NODE_PREVALENCE );
        }
    }
}

void Census::populationFromJsonFile( const std::string& filename )
{
    Demographics demographics;
    demographics.FromFile( filename );

    uint32_t defaultPopulation = 0;
    if ( demographics.HasMember( "Defaults" ) &&
         demographics[ "Defaults" ].HasMember( "NodeAttributes" ) &&
         demographics[ "Defaults" ][ "NodeAttributes" ].HasMember( "InitialPopulation" ) )
    {
        defaultPopulation = uint32_t(demographics[ "Defaults" ][ "NodeAttributes" ][ "InitialPopulation" ].GetInt64());
    }

    if ( demographics.HasMember( "Nodes" ) )
    {
        auto& nodes = demographics[ "Nodes" ];
        if ( nodes.IsArray() )
        {
            for ( size_t i = 0; i < nodes.Size(); ++i )
            {
                auto& node = nodes[ i ];
                if ( node.HasMember( "NodeAttributes" ) &&
                     node[ "NodeAttributes" ].HasMember( "InitialPopulation" ) )
                {
                    uint32_t node_population = uint32_t(node["NodeAttributes"][ "InitialPopulation" ].GetInt64());
                    if ( node_population > 0 )
                    {
                        population.push_back( node_population );
                    }
                    else
                    {
                        std::cerr << "Initial population of node " << i << " <= 0." << std::endl;
                        exit(-1);
                    }
                }
                else
                {
                    std::cerr << "Using default population " << defaultPopulation << " for node " << i << "." << std::endl;
                    population.push_back( defaultPopulation );
                }
            }
        }
        else
        {
            std::cerr << "\"Nodes\" section in demographics files is not an array." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Demographics files does not contain a \"Nodes\" section." << std::endl;
        exit(-1);
    }
}

IDistribution* distributionForObject( const rapidjson::Value& object, const std::string& element, const std::string& path = "", IDistribution* defaultDistribution = nullptr );

void Census::prevalenceFromJsonFile( const std::string& filename )
{
    Demographics demographics;
    demographics.FromFile( filename );

    IDistribution* defaultPrevalence = distributionForObject( demographics, "PrevalenceDistribution", "/Defaults/IndividualAttributes" );

    if ( demographics.HasMember( "Nodes" ) )
    {
        auto& nodes = demographics[ "Nodes" ];
        if ( nodes.IsArray() )
        {
            for ( size_t i = 0; i < nodes.Size(); ++i )
            {
                auto& node = nodes[ i ];
                auto distribution = distributionForObject( node, "PrevalenceDistribution", "/IndividualAttributes", defaultPrevalence );
                prevalence.push_back( distribution->draw() );
            }
        }
        else
        {
            std::cerr << "\"Nodes\" section in demographics files is not an array." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Demographics files does not contain a \"Nodes\" section." << std::endl;
        exit(-1);
    }
}

bool keyExists( const rapidjson::Value& object, const std::string& path );
const rapidjson::Value& getChild( const rapidjson::Value& object, const std::string& path );

IDistribution* distributionForObject( const rapidjson::Value& object, const std::string& element, const std::string& path, IDistribution* defaultDistribution )
{
    IDistribution* distribution = nullptr;
    std::string fullPath( path + "/" + element + "Flag" );
    if ( keyExists( object, fullPath ) )
    {
        auto& child = getChild( object, path );
        uint32_t type = child[ element + "Flag" ].GetUint();
        float param1 = 0.0f;
        float param2 = 0.0f;
        switch ( type )
        {
            case 1: // DISTRIBUTION_UNIFORM
            case 2: // DISTRIBUTION_GAUSSIAN
                // Needs both parameters
                if ( !child.HasMember( element + "2" ) )
                {
                    std::cerr << '"' << element.c_str() << "Flag\": " << distribution << " requires " << element.c_str() << "2 value." << std::endl;
                    exit(-1);
                }
                param2 = child[  element + "2" ].GetFloat();
                // drop through

            case 0: // DISTRIBUTION_CONSTANT
            case 3: // DISTRIBUTION_EXPONENTIAL
            case 4: // DISTRIBUTION_POISSON
                if ( !child.HasMember( element + "1" ) )
                {
                    std::cerr << '"' << element.c_str() << "Flag\": " << distribution << " requires " << element.c_str() << "1 value." << std::endl;
                    exit(-1);
                }
                param1 = child[ element + "1" ].GetFloat();
                distribution = DistributionFactory::Create( DISTRIBUTION_TYPE(type), param1, param2 );
                break;

            case 5: // DISTRIBUTION_LOGNORMAL
            case 6: // DISTRIBUTION_BIMODAL
            case 7: // DISTRIBUTION_WEIBULL
            default:
                std::cerr << "Unsupported distribution \"" << path.c_str() << '/' << element.c_str() << "Flag\": " << distribution << std::endl;
                exit(-1);
                break;
        }
    }

    if ( !distribution )
    {
        if ( defaultDistribution )
        {
            distribution = defaultDistribution; // Need some reference counting... :(
        }
        else
        {
            distribution = DistributionFactory::Create( DISTRIBUTION_TYPE::DISTRIBUTION_CONSTANT, 1.0f/128 );   // ~1%
        }
    }

    return distribution;
}

bool keyExists( const rapidjson::Value& object, const std::string& path )
{
    bool exists = (rapidjson::Pointer( path ).Get( object ) != nullptr);
    return exists;
}

const rapidjson::Value& getChild( const rapidjson::Value& object, const std::string& path )
{
    const rapidjson::Value& child = (*rapidjson::Pointer( path ).Get( object ));
    return child;
}

void validatePrevalence( const std::vector<float>& prevalence, uint32_t num_locales )
{
    if ( prevalence.size() != num_locales )
    {
        std::cerr << "Internal error, number of prevalence entries (" << prevalence.size() << ") does not match number of population nodes " << num_locales << std::endl;
        exit( -1 );
    }

    auto in_range = [] (float f) -> bool { return ((f < 0.0f) || (f > 1.0f)); };
    if ( std::count_if( prevalence.begin(), prevalence.end(), in_range ) > 0 )
    {
        std::cerr << "Found out-of-range (0 <= prevalence <= 1) value(s) in prevalence file." << std::endl;
        exit( -1 );
    }
}

/*
void censusFromBinaryFile( const std::string& filename, std::vector<uint32_t>& population )
{
    std::cerr << "Loading binary population file '" << filename.c_str() << "'..." << std::endl;
    auto file_size = CheckFileSize( filename );
    FILE* file = fopen( filename.c_str(), "rb" );
    if ( file != nullptr )
    {
        size_t num_locations = file_size / sizeof(uint32_t);
        population.resize( num_locations );
        std::cerr << "Reading population data for " << num_locations << " nodes..." << std::endl;
        auto count = fread( population.data(), sizeof(uint32_t), num_locations, file );
        std::cerr << "Read population data for " << count << " nodes." << std::endl;
        fclose( file );
        if ( count != num_locations )
        {
            std::cerr << "Error reading population data from '" << filename.c_str() << "'." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Couldn't open " << filename.c_str() << " for reading population data." << std::endl;
        exit(-1);
    }
}

void censusFromCsvFile( const std::string& filename, std::vector<uint32_t>& population )
{
    std::cerr << "Loading CSV population file '" << filename.c_str() << "'..." << std::endl;
    io::CSVReader<1> in(filename);
    in.read_header(io::ignore_extra_column, "population");
    uint32_t population;
    while( in.read_row(population) )
    {
        population.push_back( population );
    }
}

void prevalenceFromBinaryFile( const std::string& filename, std::vector<float>& prevalences )
{
    std::cerr << "Loading binary prevalence file '" << filename.c_str() << "'..." << std::endl;
    auto file_size = CheckFileSize( filename );
    FILE* file = fopen( filename.c_str(), "rb" );
    if ( file != nullptr )
    {
        size_t num_locations = file_size / sizeof(float);
        prevalences.resize( num_locations );
        std::cerr << "Reading prevalence data for " << num_locations << " nodes..." << std::endl;
        auto count = fread( prevalences.data(), sizeof(float), num_locations, file );
        std::cerr << "Read prevalence data for " << count << " nodes." << std::endl;
        fclose( file );
        if ( count != num_locations )
        {
            std::cerr << "Error reading prevalence data from '" << filename.c_str() << "'." << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "Couldn't open " << filename.c_str() << " for reading prevalence data." << std::endl;
        exit(-1);
    }
}

void prevalenceFromCsvFile( const std::string& filename, std::vector<float>& prevalences )
{
    std::cerr << "Loading CSV prevalence file '" << filename.c_str() << "'..." << std::endl;
    io::CSVReader<1> in(filename);
    in.read_header(io::ignore_extra_column, "prevalence");
    float prevalence;
    while( in.read_row(prevalence) )
    {
        prevalences.push_back( prevalence );
    }
}
*/