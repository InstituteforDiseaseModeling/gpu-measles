/*
** copyright
*/

#pragma once

#include <string>

namespace util
{
    enum FileType {
        typeBinary = 0,
        typeCsv    = 1,
        typeJson   = 2
    };

    FileType DetermineFileType( const std::string& filename );
}
