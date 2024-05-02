/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#ifndef LASER_H_
#define LASER_H_

#include <AMReX_Vector.H>
#include <AMReX_RealVect.H>
#include "utils/Constants.H"

class Laser
{
public:

    Laser (std::string name, bool laser_from_file);

    std::string m_name {""};
    amrex::Real m_a0 {0.}; /**< Laser peak normalized amplitude */
    amrex::Real m_w0 {0.}; /**< Laser waist */
    amrex::Real m_CEP {0.}; /**< Laser carrier-envelope phase (CEP) */
    /** Propagation angle of the pulse in the yz plane (0 is the along the z axis) */
    amrex::Real m_propagation_angle_yz {0.};
    /**Pulse front tilt angle of the pulse in yz plane (pi/2 is no PFT)*/
    amrex::Real m_PFT_yz {MathConst::pi/2.0};
    amrex::Real m_L0 {0.}; /**< Laser length (HW 1/e in amplitude) */
    amrex::Real m_tau {0.}; /**< Laser duration (HW 1/e in amplitude) */
    /** Focal distance of the laser pulse */
    amrex::Real m_focal_distance {0.};
    /** Average position of the Gaussian laser pulse */
    amrex::RealVect m_position_mean {0., 0., 0.};
};

#endif // LASER_H_
