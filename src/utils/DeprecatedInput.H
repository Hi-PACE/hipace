/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_DeprecatedInput_H_
#define HIPACE_DeprecatedInput_H_

#include "Parser.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

inline void
DeprecatedInput (std::string const& pp_name, char const * const str,
                std::string const& replacement, std::string const& msg="", bool new_prefix=false)
{
    amrex::ParmParse pp(pp_name);
    std::string tmp_str = "";
    if (queryWithParser(pp, str, tmp_str)) {
        amrex::Abort(
            "DEPRECATED INPUT ERROR:\n"
            "Input parameter " + pp_name + "." + str + " no longer supported.\n"
            "See " + (new_prefix ? "" : pp_name + ".") + replacement +
            " instead (more info in the documentation). " + msg);
    }
}

#endif // HIPACE_DeprecatedInput_H_
