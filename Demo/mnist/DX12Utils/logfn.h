///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

enum class LogLevel : int
{
    Info,
    Warn,
    Error
};
using TLogFn = void (*)(LogLevel level, const char* msg, ...);
