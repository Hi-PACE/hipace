/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "GetInitialDensity.H"
#include "utils/Parser.H"

GetInitialDensity::GetInitialDensity (const std::string& name, amrex::Parser& parser)
{
    amrex::ParmParse pp(name);
    std::string profile;
    getWithParser(pp, "profile", profile);

    if (profile == "gaussian") {
        m_profile = BeamProfileType::Gaussian;
        getWithParser(pp, "density", m_density);
        m_density = std::abs(m_density);
        queryWithParser(pp, "position_mean", m_position_mean);
        queryWithParser(pp, "position_std", m_position_std);
    } else if (profile == "flattop") {
        m_profile = BeamProfileType::Flattop;
        getWithParser(pp, "density", m_density);
        m_density = std::abs(m_density);
    } else if (profile == "parsed") {
        m_profile = BeamProfileType::Parsed;
        std::string density_func_str = "0.";
        getWithParser(pp, "density(x,y,z)", density_func_str);
        m_density_func = makeFunctionWithParser<3>(density_func_str, parser, {"x", "y", "z"});
    } else {
        amrex::Abort("Unknown beam profile!");
    }
}
