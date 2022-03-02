/* Copyright 2020-2021 AlexanderSinn, MaxThevenet, Severin Diederichs
 *
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "GetInitialDensity.H"
#include "utils/Parser.H"

GetInitialDensity::GetInitialDensity (const std::string& name)
{
    amrex::ParmParse pp(name);
    std::string profile;
    getWithParser(pp, "density", m_density);
    getWithParser(pp, "profile", profile);

    if        (profile == "gaussian") {
        m_profile = BeamProfileType::Gaussian;
    } else if (profile == "flattop") {
        m_profile = BeamProfileType::Flattop;
    } else {
        amrex::Abort("Unknown beam profile!");
    }

    if (m_profile == BeamProfileType::Gaussian) {
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        if (queryWithParser(pp, "position_mean", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_position_mean[idim] = loc_array[idim];
            }
        }
        if (queryWithParser(pp, "position_std", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_position_std[idim] = loc_array[idim];
            }
        }
    }
}
