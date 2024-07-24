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
    queryWithParser(pp, "init_type", m_laser_init_type);
    if (m_laser_init_type == "from_file") return;
    else if (m_laser_init_type == "gaussian"){
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
        queryWithParser(pp, "position_mean",  m_position_mean);
        return;
    }
    else if (m_laser_init_type == "parser"){
        std::string profile_real_str = "";
        std::string profile_imag_str = "";
        getWithParser(pp, "laser_real(x,y,z)", profile_real_str);
        getWithParser(pp, "laser_imag(x,y,z)", profile_imag_str);
        m_profile_real = makeFunctionWithParser<3>( profile_real_str, m_parser_lr, {"x", "y", "z"});
        m_profile_imag = makeFunctionWithParser<3>( profile_imag_str, m_parser_li, {"x", "y", "z"});
        return;
    }
}
