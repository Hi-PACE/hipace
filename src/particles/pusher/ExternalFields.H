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
 * \param[in] time time of the current step
 * \param[in] clight speed of light
 * \param[in,out] ExmByp Ex-By Field on particle
 * \param[in,out] EypBxp Ey+Bx Field on particle
 * \param[in,out] Ezp Electric field on particle, z component
 * \param[in,out] Bxp Magnetic field on particle, x component
 * \param[in,out] Byp Magnetic field on particle, y component
 * \param[in,out] Bzp Magnetic field on particle, z component
 * \param[in] external_fields External field functions for Ex Ey Ez Bx By Bz
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ApplyExternalField(
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    const amrex::ParticleReal zp,
    const amrex::ParticleReal time,
    const amrex::ParticleReal clight,
    amrex::ParticleReal& ExmByp,
    amrex::ParticleReal& EypBxp,
    amrex::ParticleReal& Ezp,
    amrex::ParticleReal& Bxp,
    amrex::ParticleReal& Byp,
    amrex::ParticleReal& Bzp,
    amrex::GpuArray<amrex::ParserExecutor<4>, 6> external_fields)
{
    const amrex::Real Ex = external_fields[0](xp, yp, zp, time);
    const amrex::Real Ey = external_fields[1](xp, yp, zp, time);
    const amrex::Real Ez = external_fields[2](xp, yp, zp, time);
    const amrex::Real Bx = external_fields[3](xp, yp, zp, time);
    const amrex::Real By = external_fields[4](xp, yp, zp, time);
    const amrex::Real Bz = external_fields[5](xp, yp, zp, time);

    ExmByp += Ex - clight * By;
    EypBxp += Ey + clight * Bx;
    Ezp    += Ez;
    Bxp    += Bx;
    Byp    += By;
    Bzp    += Bz;
}

#endif // EXTERNALFIELDS_H_
