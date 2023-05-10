/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef ROCFFTUTILS_H_
#define ROCFFTUTILS_H_

#if __has_include(<rocfft/rocfft.h>)  // ROCm 5.3+
#  include <rocfft/rocfft.h>
#else
#  include <rocfft.h>
#endif

#include <string>


namespace RocFFTUtils
{
    void assert_rocfft_status (std::string const& name, rocfft_status status);

    /** \brief This method converts a cufftResult
     * into the corresponding string
     *
     * @param[in] err a cufftResult
     * @return an std::string
     */
    std::string rocfftErrorToString (const rocfft_status err);
}

#endif // ROCFFTUTILS_H_
