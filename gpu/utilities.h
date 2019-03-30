/*
** copyright
*/

#pragma once

#include <ctime>
#include <iostream>
#include <string>

namespace util
{
    enum FileType {
        typeBinary = 0,
        typeCsv    = 1,
        typeJson   = 2
    };

    FileType DetermineFileType( const std::string& filename );
    size_t CheckFileSize( const std::string& filename );

    void PrintElapsedTime(clock_t start, clock_t end, const char* msg = nullptr, std::ostream& stream = std::cerr);
}
