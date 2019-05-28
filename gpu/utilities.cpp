/*
** copyright
*/

#include <algorithm>
#include <iostream>
#include <sys/stat.h>

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

    size_t CheckFileSize( const std::string& filename )
    {
        struct stat file_stats;
    
        stat( filename.c_str(), &file_stats );
    
        if ( file_stats.st_size > 0 )
        {
            if ( (file_stats.st_size % sizeof(uint32_t)) == 0 )
            {
                // all is good, nothing to do
            }
            else
            {
                std::cerr << "File '" << filename.c_str() << "' is not an even number of uint32 values." << std::endl;
                exit(-1);
            }
        }
        else
        {
            std::cerr << "File '" << filename.c_str() << "' is empty." << std::endl;
            exit(-1);
        }
    
        return file_stats.st_size;
    }
    
    void PrintElapsedTime(clock_t start, clock_t end, const char* msg, std::ostream& stream)
    {
        if ( msg )
        {
            stream << msg << " : ";
        }
        stream << (1000 * uint64_t(end - start) / double(CLOCKS_PER_SEC)) << " milliseconds" << std::endl;
    }
}