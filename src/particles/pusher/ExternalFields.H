/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef EXTERNALFIELDS_H_
#define EXTERNALFIELDS_H_

/** \brief add axisymmetric linear focusing field on particles and linear accelerating field.
 * Typically called right after
 * the field gather.
 *
 * \param[in] xp particle position x
 * \param[in] yp particle position y
 * \param[in] zp particle position x
 * \param[in,out] ExmByp Ex-By Field on particle
 * \param[in,out] EypBxp Ey+Bx Field on particle
 * \param[in,out] Ezp Ez Field on particle
 * \param[in] ext_E_uniform Uniform electric field Ex-c*By, Ey+c*Bx, Ez to be applied
 * \param[in] ext_B_uniform Uniform magnetic field Bx, B, Bz to be applied
 * \param[in] ext_E_slope strength of external electric field d(Ex-c*By)/dx, d(Ey+c*Bx)/dy, d(Ez)/dz
 * \param[in] ext_B_slope strength of external magnetic field d(Bx)/dy, d(By)/dx, d(Bz)/dz
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ApplyExternalField(
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    const amrex::ParticleReal zp,
    amrex::ParticleReal& ExmByp,
    amrex::ParticleReal& EypBxp,
    amrex::ParticleReal& Ezp,
    amrex::ParticleReal& Bxp,
    amrex::ParticleReal& Byp,
    amrex::ParticleReal& Bzp,
    const amrex::RealVect& ext_E_uniform,
    const amrex::RealVect& ext_B_uniform,
    const amrex::RealVect& ext_E_slope,
    const amrex::RealVect& ext_B_slope)
{

    ExmByp += ext_E_uniform[0] + ext_E_slope[0] * xp;
    EypBxp += ext_E_uniform[1] + ext_E_slope[1] * yp;
    Ezp    += ext_E_uniform[2] + ext_E_slope[2] * zp;
    Bxp    += ext_B_uniform[0] + ext_B_slope[0] * yp;
    Byp    += ext_B_uniform[1] + ext_B_slope[1] * xp;
    Bzp    += ext_B_uniform[2] + ext_B_slope[2] * zp;

    return;
}

#endif // EXTERNALFIELDS_H_
