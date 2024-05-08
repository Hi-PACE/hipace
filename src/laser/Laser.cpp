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

Laser::Laser (std::string name, bool laser_from_file)
{
    m_name = name;
    if (laser_from_file) return;
    amrex::ParmParse pp(m_name);
    queryWithParser(pp, "a0", m_a0);
    queryWithParser(pp, "w0", m_w0);
    queryWithParser(pp, "CEP", m_CEP);
    queryWithParser(pp, "propagation_angle_yz", m_propagation_angle_yz);
    queryWithParser(pp, "PFT_yz", m_PFT_yz);
    bool length_is_specified = queryWithParser(pp, "L0", m_L0);
    bool duration_is_specified = queryWithParser(pp, "tau", m_tau);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( length_is_specified + duration_is_specified == 1,
        "Please specify exlusively either the pulse length L0 or the duration tau of the laser");
    if (duration_is_specified) m_L0 = m_tau*get_phys_const().c;
    queryWithParser(pp, "focal_distance", m_focal_distance);

    queryWithParser(pp, "position_mean", m_position_mean);
}
