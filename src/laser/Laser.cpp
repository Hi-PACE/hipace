/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#include "Laser.H"
#include "utils/Parser.H"
#include "Hipace.H"

#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

Laser::Laser (std::string name)
{
    m_name = name;
    amrex::ParmParse pp(m_name);
    queryWithParser(pp, "a0", m_a0);
    queryWithParser(pp, "w0", m_w0);
    bool length_is_specified = queryWithParser(pp, "L0", m_L0);;
    bool duration_is_specified = queryWithParser(pp, "tau", m_tau);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( length_is_specified + duration_is_specified == 1,
        "Please specify exlusively either the pulse length L0 or the duration tau of the laser");
    if (duration_is_specified) m_L0 = m_tau/get_phys_const().c;
    queryWithParser(pp, "focal_distance", m_focal_distance);

    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    queryWithParser(pp, "position_mean", loc_array);
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
}
