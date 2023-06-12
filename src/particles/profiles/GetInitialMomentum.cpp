/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "GetInitialMomentum.H"
#include "utils/Parser.H"

GetInitialMomentum::GetInitialMomentum (const std::string& name)
{
    amrex::ParmParse pp(name);

    /* currently only Gaussian beam momentum profile implemented */
    if (m_momentum_profile == BeamMomentumType::Gaussian) {

        queryWithParser(pp, "u_mean", m_u_mean);
        queryWithParser(pp, "u_std", m_u_std);
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
