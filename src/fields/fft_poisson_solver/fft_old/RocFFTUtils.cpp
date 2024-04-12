/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "RocFFTUtils.H"
#include <AMReX.H>

#include <map>
#include <string>

namespace RocFFTUtils
{
    void assert_rocfft_status (std::string const& name, rocfft_status status)
    {
        if (status != rocfft_status_success) {
            amrex::Abort(name + " failed! Error: " + rocfftErrorToString(status));
        }
    }

    std::string rocfftErrorToString (const rocfft_status err)
    {
        if              (err == rocfft_status_success) {
            return std::string("rocfft_status_success");
        } else if       (err == rocfft_status_failure) {
            return std::string("rocfft_status_failure");
        } else if       (err == rocfft_status_invalid_arg_value) {
            return std::string("rocfft_status_invalid_arg_value");
        } else if       (err == rocfft_status_invalid_dimensions) {
            return std::string("rocfft_status_invalid_dimensions");
        } else if       (err == rocfft_status_invalid_array_type) {
            return std::string("rocfft_status_invalid_array_type");
        } else if       (err == rocfft_status_invalid_strides) {
            return std::string("rocfft_status_invalid_strides");
        } else if       (err == rocfft_status_invalid_distance) {
            return std::string("rocfft_status_invalid_distance");
        } else if       (err == rocfft_status_invalid_offset) {
            return std::string("rocfft_status_invalid_offset");
        } else {
            return std::to_string(err) + " (unknown error code)";
        }
    }
}
