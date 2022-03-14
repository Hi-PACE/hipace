/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Hipace.H"
#include "HipaceVersion.H"

#include <string>


std::string
Hipace::Version ()
{
#ifdef HIPACE_GIT_VERSION
    return std::string(HIPACE_GIT_VERSION);
#else
    return std::string("Unknown");
#endif
}
