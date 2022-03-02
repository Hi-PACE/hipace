/* Copyright 2020-2021 AlexanderSinn, MaxThevenet, Severin Diederichs
 *
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "GetInitialMomentum.H"
#include "utils/Parser.H"

GetInitialMomentum::GetInitialMomentum (const std::string& name)
{
    amrex::ParmParse pp(name);

    /* currently only Gaussian beam momentum profile implemented */
    if (m_momentum_profile == BeamMomentumType::Gaussian) {

        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        if (queryWithParser(pp, "u_mean", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_u_mean[idim] = loc_array[idim];
            }
        }
        if (queryWithParser(pp, "u_std", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_u_std[idim] = loc_array[idim];
            }
        }
        bool do_symmetrize = false;
        queryWithParser(pp, "do_symmetrize", do_symmetrize);
        if (do_symmetrize) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE( std::fabs(m_u_mean[0]) +std::fabs(m_u_mean[1])
                                               < std::numeric_limits<amrex::Real>::epsilon(),
            "Symmetrizing the beam is only implemented for no mean momentum in x and y");
        }
    } else {
        amrex::Abort("Unknown beam momentum profile!");
    }
}
