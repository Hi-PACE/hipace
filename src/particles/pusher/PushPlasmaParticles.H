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
    T dz_psi_inv;
};

/** \brief Find the derivative of the Plasma momentum.
 * This function can be used with dual numbers to find the second derivative.
 *
 * \param[in] ux gamma * beta_x * clight
 * \param[in] uy gamma * beta_y * clight
 * \param[in] psi_inv 1 / (1 + norm_Psi)
 * \param[in] ExmBy Ex - clight * By
 * \param[in] EypBx Ey - clight * Bx
 * \param[in] Ez Ez
 * \param[in] Bx_clight Bx * clight
 * \param[in] By_clight By * clight
 * \param[in] Bz Bz
 * \param[in] Aabssq_norm Aabssqp * 0.5 TODO: fix units of Aabssqp
 * \param[in] AabssqDx_norm AabssqDx * 0.25 * clight * m_e / mass
 * \param[in] AabssqDy_norm AabssqDy * 0.25 * clight * m_e / mass
 * \param[in] clight_inv 1 / clight
 * \param[in] charge_mass_clight_ratio charge / (mass * clight)
 */

template<class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PlasmaMomentumDerivative<T> PlasmaMomentumPush (
    const T& ux,
    const T& uy,
    const T& psi_inv,
    amrex::Real ExmBy,
    amrex::Real EypBx,
    amrex::Real Ez,
    amrex::Real Bx_clight,
    amrex::Real By_clight,
    amrex::Real Bz,
    amrex::Real Aabssq_norm,
    amrex::Real AabssqDx_norm,
    amrex::Real AabssqDy_norm,
    amrex::Real clight_inv,
    amrex::Real charge_mass_clight_ratio)
{
    using namespace amrex::literals;

    const T gamma_psi = 0.5_rt*psi_inv*psi_inv*(
                        1.0_rt + Aabssq_norm
                        + ux*ux*(clight_inv*clight_inv)
                        + uy*uy*(clight_inv*clight_inv))
                        + 0.5_rt;

    const T dz_ux = (charge_mass_clight_ratio * (gamma_psi * ExmBy
                    + By_clight + ( uy * Bz ) * psi_inv) - AabssqDx_norm * psi_inv);

    const T dz_uy = (charge_mass_clight_ratio * (gamma_psi * EypBx
                    - Bx_clight - ( ux * Bz ) * psi_inv) - AabssqDy_norm * psi_inv);

    const T dz_psi_inv = psi_inv * psi_inv * (-charge_mass_clight_ratio * clight_inv *
                         (( ux * ExmBy + uy * EypBx ) * clight_inv * psi_inv - Ez ));

    return {dz_ux, dz_uy, dz_psi_inv};
}

#endif //  PUSHPLASMAPARTICLES_H_
