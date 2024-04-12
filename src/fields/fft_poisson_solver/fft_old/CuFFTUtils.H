/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef CUFFTUTILS_H_
#define CUFFTUTILS_H_

#include <cufft.h>

#include <string>


namespace CuFFTUtils
{
    /** \brief This method converts a cufftResult
     * into the corresponding string
     *
     * @param[in] err a cufftResult
     * @return an std::string
     */
    std::string cufftErrorToString (const cufftResult& err);
}

#endif // CUFFTUTILS_H_
