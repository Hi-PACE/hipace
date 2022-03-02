/* Copyright 2020 MaxThevenet
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CuFFTUtils.H"

#include <map>

namespace CuFFTUtils
{
    std::string cufftErrorToString (const cufftResult& err)
    {
        const auto res2string = std::map<cufftResult, std::string>{
            {CUFFT_SUCCESS, "CUFFT_SUCCESS"},
            {CUFFT_INVALID_PLAN,"CUFFT_INVALID_PLAN"},
            {CUFFT_ALLOC_FAILED,"CUFFT_ALLOC_FAILED"},
            {CUFFT_INVALID_TYPE,"CUFFT_INVALID_TYPE"},
            {CUFFT_INVALID_VALUE,"CUFFT_INVALID_VALUE"},
            {CUFFT_INTERNAL_ERROR,"CUFFT_INTERNAL_ERROR"},
            {CUFFT_EXEC_FAILED,"CUFFT_EXEC_FAILED"},
            {CUFFT_SETUP_FAILED,"CUFFT_SETUP_FAILED"},
            {CUFFT_INVALID_SIZE,"CUFFT_INVALID_SIZE"},
            {CUFFT_UNALIGNED_DATA,"CUFFT_UNALIGNED_DATA"}};

        const auto it = res2string.find(err);
        if(it != res2string.end()){
            return it->second;
        }
        else{
            return std::to_string(err) +
                " (unknown error code)";
        }
    }
}
