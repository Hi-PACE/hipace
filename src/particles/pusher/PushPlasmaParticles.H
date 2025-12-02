/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef PUSHPLASMAPARTICLES_H_
#define PUSHPLASMAPARTICLES_H_

#include <AMReX_Config.H>

template<class T>
struct PlasmaMomentumDerivative {
    T dz_ux;
    T dz_uy;
    T dz_psi;
};

/** \brief Find the derivative of the Plasma momentum.
 * This function can be used with dual numbers to find the second derivative.
 *
 * \param[in] ux gamma * beta_x
 * \param[in] uy gamma * beta_y
 * \param[in] psi_inv 1 / (1 + norm_Psi)
 * \param[in] ExmBy_ci Ex / clight - By
 * \param[in] EypBx_ci Ey / clight + Bx
 * \param[in] Ez_ci Ez / clight
 * \param[in] Bx Bx
 * \param[in] By By
 * \param[in] Bz Bz
 * \param[in] Aabssq_norm Aabssqp * 0.5 * (charge / q_e)^2 * (m_e / mass)^2
 * \param[in] AabssqDx_norm AabssqDx * 0.25  * (charge / q_e)^2 * (m_e / mass)^2
 * \param[in] AabssqDy_norm AabssqDy * 0.25  * (charge / q_e)^2 * (m_e / mass)^2
 * \param[in] charge_mass_clight_ratio charge / (mass * clight)
 */

template<class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PlasmaMomentumDerivative<T> PlasmaMomentumPush (
    const T& ux,
    const T& uy,
    const T& psi_inv,
    amrex::Real ExmBy_ci,
    amrex::Real EypBx_ci,
    amrex::Real Ez_ci,
    amrex::Real Bx,
    amrex::Real By,
    amrex::Real Bz,
    amrex::Real Aabssq_norm,
    amrex::Real AabssqDx_norm,
    amrex::Real AabssqDy_norm,
    amrex::Real charge_mass_clight_ratio)
{
    using namespace amrex::literals;

    const T gamma_psi = 0.5_rt*psi_inv*psi_inv*(
                        1.0_rt + Aabssq_norm
                        + ux*ux
                        + uy*uy)
                        + 0.5_rt;

    const T dz_ux = (charge_mass_clight_ratio * (gamma_psi * ExmBy_ci
                    + By + ( uy * Bz ) * psi_inv) - AabssqDx_norm * psi_inv);

    const T dz_uy = (charge_mass_clight_ratio * (gamma_psi * EypBx_ci
                    - Bx - ( ux * Bz ) * psi_inv) - AabssqDy_norm * psi_inv);

    const T dz_psi = (charge_mass_clight_ratio *
                         (( ux * ExmBy_ci + uy * EypBx_ci ) * psi_inv - Ez_ci ));

    return {dz_ux, dz_uy, dz_psi};
}

#endif //  PUSHPLASMAPARTICLES_H_
