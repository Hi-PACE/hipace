/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_ELASTIC_COLLISION_PEREZ_H_
#define HIPACE_ELASTIC_COLLISION_PEREZ_H_

#include "utils/Constants.H"
#include "UpdateMomentumPerez.H"
#include "ComputeTemperature.H"

#include <AMReX_Random.H>

/** \brief Prepare information for and call
 *        UpdateMomentumPerezElastic().
 * @param[in] I1s start index for I1 (inclusive).
 * @param[in] I2s start index for I2 (inclusive).
 * @param[in] I1e start index for I1 (exclusive).
 * @param[in] I2e start index for I2 (exclusive).
 * @param[in] I1 index array.
 * @param[in] I2 index array.
 * @param[in,out] u1x x Proper velocity (u=v*gamma) array of species 1.
 * @param[in,out] u1y y Proper velocity (u=v*gamma) array of species 1.
 * @param[in,out] psi1 pseudo-potential array of species 1.
 * @param[in,out] u2x x Proper velocity (u=v*gamma) array of species 2.
 * @param[in,out] u2y y Proper velocity (u=v*gamma) array of species 2.
 * @param[in,out] psi2 pseudo-potential array of species 2.
 * @param[in] w1 array of weights.
 * @param[in] w2 array of weights.
 * @param[in] q1 Physical charge of species 1.
 * @param[in] q2 Physical charge of species 2.
 * @param[in] m1 Physical mass of species 1.
 * @param[in] m2 Physical mass of species 2.
 * @param[in] T1 temperature of species 1 (Joule). If <0, measured per-cell.
 * @param[in] T2 temperature of species 2 (Joule). If <0, measured per-cell.
 * @param[in] dt is the time step length between two collision calls.
 * @param[in] L Coulomb log. If <0, measured per cell.
 * @param[in] dV Volume of the corresponding cell.
 * @param[in] engine AMReX engine for the random number generator.
*/

template <typename T_index, typename T_R>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ElasticCollisionPerez (
    T_index const I1s, T_index const I1e,
    T_index const I2s, T_index const I2e,
    T_index *I1,       T_index *I2,
    T_R *u1x, T_R *u1y, T_R *psi1,
    T_R *u2x, T_R *u2y, T_R *psi2,
    T_R const *w1, T_R const *w2,
    T_R const  q1, T_R const  q2,
    T_R const  m1, T_R const  m2,
    T_R const  T1, T_R const  T2,
    T_R const  dt, T_R const   L, T_R const dV,
    amrex::RandomEngine const& engine)
{
    using namespace amrex::literals;

    T_R constexpr inv_c2 = T_R(1.0)/(PhysConstSI::c*PhysConstSI::c);
    const int NI1 = I1e - I1s;
    const int NI2 = I2e - I2s;

    // get local T1t and T2t
    T_R T1t; T_R T2t;
    if ( T1 <= T_R(0.0) && L <= T_R(0.0) )
    {
        T1t = ComputeTemperature(I1s,I1e,I1,u1x,u1y,psi1,m1);
    }
    else { T1t = T1; }
    if ( T2 <= T_R(0.0) && L <= T_R(0.0) )
    {
        T2t = ComputeTemperature(I2s,I2e,I2,u2x,u2y,psi2,m2);
    }
    else { T2t = T2; }

    // local density
    T_R n1  = T_R(0.0);
    T_R n2  = T_R(0.0);
    T_R n12 = T_R(0.0);
    for (int i1=I1s; i1<static_cast<int>(I1e); ++i1) { n1 += w1[ I1[i1] ]; }
    for (int i2=I2s; i2<static_cast<int>(I2e); ++i2) { n2 += w2[ I2[i2] ]; }
    n1 = n1 / dV; n2 = n2 / dV;
    {
      int i1 = I1s; int i2 = I2s;
      for (int k = 0; k < amrex::max(NI1,NI2); ++k)
      {
        n12 += amrex::min( w1[ I1[i1] ], w2[ I2[i2] ] );
        ++i1; if ( i1 == static_cast<int>(I1e) ) { i1 = I1s; }
        ++i2; if ( i2 == static_cast<int>(I2e) ) { i2 = I2s; }
      }
      n12 = n12 / dV;
    }

    // compute Debye length lmdD
    T_R lmdD;
    if ( T1t < T_R(0.0) || T2t < T_R(0.0) ) {
        lmdD = T_R(0.0);
    }
    else {
        lmdD = T_R(1.0)/std::sqrt( n1*q1*q1/(T1t*PhysConstSI::ep0) +
                                   n2*q2*q2/(T2t*PhysConstSI::ep0) );
    }
    T_R rmin = std::pow( T_R(4.0) * MathConst::pi / T_R(3.0) *
               amrex::max(n1,n2), T_R(-1.0/3.0) );
    lmdD = amrex::max(lmdD, rmin);

    // call UpdateMomentumPerezElastic()
    {
        int i1 = I1s; int i2 = I2s;
        for (int k = 0; k < amrex::max(NI1,NI2); ++k)
        {
            // particle's normalized vector potential
            const amrex::Real psi1i = psi1[I1[i1]] *
                PhysConstSI::q_e / (PhysConstSI::m_e * PhysConstSI::c * PhysConstSI::c);
            // particle's Lorentz factor
            const amrex::Real g1 = (
                1.0_rt +
                u1x[I1[i1]]*u1x[I1[i1]]*inv_c2 + u1y[I1[i1]]*u1y[I1[i1]]*inv_c2 +
                (psi1i+1._rt)*(psi1i+1._rt)) / (2.0_rt * (psi1i+1._rt) );
            // particle's normalized vector potential
            const amrex::Real psi2i = psi2[I2[i2]] *
                PhysConstSI::q_e / (PhysConstSI::m_e * PhysConstSI::c * PhysConstSI::c);
            // particle's Lorentz factor
            const amrex::Real g2 = (
                1.0_rt +
                u2x[I2[i2]]*u2x[I2[i2]]*inv_c2 + u2y[I2[i2]]*u2y[I2[i2]]*inv_c2 +
                (psi2i+1._rt)*(psi2i+1._rt)) / (2.0_rt * (psi2i+1._rt) );
            // In the longitudinal push of plasma particles, the dt is different for each particle.
            // The dt applied for collision probability is the average (in the lab frame) of these
            // dts. This is NOT clean. TODO FIXME.
            const amrex::Real dt_fac = 0.5_rt * (g1/(psi1i+1._rt) + g2/(psi2i + 1._rt));
            UpdateMomentumPerezElastic(
                u1x[ I1[i1] ], u1y[ I1[i1] ], psi1[ I1[i1] ],
                u2x[ I2[i2] ], u2y[ I2[i2] ], psi2[ I2[i2] ],
                n1, n2, n12,
                q1, m1, w1[ I1[i1] ], q2, m2, w2[ I2[i2] ],
                dt * dt_fac, L, lmdD, engine);

            ++i1; if ( i1 == static_cast<int>(I1e) ) { i1 = I1s; }
            ++i2; if ( i2 == static_cast<int>(I2e) ) { i2 = I2s; }
        }
    }
}

#endif // HIPACE_ELASTIC_COLLISION_PEREZ_H_
