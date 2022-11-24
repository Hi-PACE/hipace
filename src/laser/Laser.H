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

class Laser
{
public:
    Laser (std::string name);
/*
    amrex::Real getA0 () {return m_a0};
    amrex::Real getK0 () {return m_k0};
    amrex::Real getW0 () {return m_w0};
    amrex::Real getL0 () {return m_l0};
    amrex::Real getTau () {return m_tau};
    amrex::Real getLambda () {return m_lambda0};
    amrex::Real getFoc () {return m_focal_distance};
    amrex::Real getX0 () {return m_position_mean[0]};
    amrex::Real getY0 () {return m_position_mean[1]};
    amrex::Real getZ0 () {return m_position_mean[2]};
*/
    std::string m_name {""};
    amrex::Real m_a0 {0.}; /**< Laser peak normalized amplitude */
    amrex::Real m_w0 {0.}; /**< Laser waist */
    amrex::Real m_L0 {0.}; /**< Laser length (HW 1/e in amplitude) */
    amrex::Real m_tau {0.}; /**< Laser duration (HW 1/e in amplitude) */
    /** Focal distance of the laser pulse */
    amrex::Real m_focal_distance {0.};
    /** Average position of the Gaussian laser pulse */
    amrex::RealVect m_position_mean {0., 0., 0.};
};

#endif // LASER_H_
