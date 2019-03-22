/*
** copyright
*/

#include "utilities.h"

namespace util
{
    FileType DetermineFileType( const std::string& filename )
    {
        FileType type = typeBinary;   // default

        auto dot = filename.find_last_of(".");

        if ( dot > 0 )
        {
            std::string suffix = filename.substr( dot + 1 );
            std::transform( suffix.begin(), suffix.end(), suffix.begin(), ::tolower );

            if ( suffix == std::string("csv") )
            {
                type = typeCsv;
                std::cerr << "'" << filename.c_str() << "' looks to be a CSV file." << std::endl << std::flush;
            }
            else if ( suffix == std::string("json") )
            {
                type = typeJson;
                std::cerr << "'" << filename.c_str() << "' looks to be a JSON file." << std::endl << std::flush;
            }
            else
            {
                // do nothing, use default, typeBinary
                std::cerr << "'" << filename.c_str() << "' looks to be a binary file." << std::endl << std::flush;
            }
        }
        else
        {
            // do nothing, use default, typeBinary
        }

        return type;
    }
}