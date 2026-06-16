/* This file has been derived from
 * https://github.com/BLAST-WarpX/warpx/blob/development/Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H
 * 
 * Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef HIPACE_IMPACT_IONIZATION_H_
#define HIPACE_IMPACT_IONIZATION_H_

#include "utils/Constants.H"

#include <AMReX_Random.H>

#include <cmath>  // isnan() isinf()
#include <limits> // numeric_limits<float>::min()

/* \brief Relativistic impact-ionization final-state model.
 *
 *        The collision is handled in the center-of-momentum (COM) frame.
 *        Ionization is allowed only if the COM kinetic energy exceeds the ionization energy.
 *
 *        After ionization, three outgoing particles are produced:
 *
 *            particle 1 : scattered projectile electron
 *            particle 2 : target ion/ionized neutral
 *            particle 3 : secondary electron
 *        The two outgoing electrons are assigned independent isotropic random directions. 
 *        Their momentum magnitudes are assumed to be equal, |p1| = |p3| = C, 
 *        which is consistent with an equal-energy-share assumption for equal-mass electrons. 
 *        The ion momentum is then fixed by momentum conservation:
 *
 *            p2 = -(p1 + p3).
 *
 *        The scalar C is determined by solving
 *
 *            K1(C) + K2(C) + K3(C) = Eout,
 *
 *        where Ki is the relativistic kinetic energy of particle i and Eout is the COM kinetic energy 
 *        remaining after subtracting the ionization energy. The root is found by bisection.
 *
 *        The model assumes that particles 1 and 3 are electrons with equal mass. 
 *        Compile with USE_ASSERTION=TRUE to enable checks for NaN or Inf updated momenta.
 */

template <typename T_R>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ImpactIonization (
    T_R& ux1, T_R& uy1, T_R& uz1, T_R const g1,
    T_R& ux2, T_R& uy2, T_R& uz2, T_R const g2, int& ion_lev2,
    T_R& ux3, T_R& uy3, T_R& uz3,
    T_R m1, T_R const w1r, T_R m2, T_R const w2r, T_R const m3,
    T_R const Eion_eV, T_R const c2, T_R const inv_c, T_R const inv_c2,
    amrex::RandomEngine const& engine)
{
    using namespace amrex::literals;

    // Tolerances for numerical checks
    constexpr T_R rel_tol  = T_R(1.0e-10);
    constexpr T_R eps_n    = T_R(1.0e-10);
    constexpr T_R beta_tol = T_R(1.0e-12);

    T_R const Eion  = Eion_eV * PhysConstSI::q_e;  // [J]

    // ------------------------------------------------------------------
    // Early-out if relative motion is negligible
    // ------------------------------------------------------------------
    T_R const diffx = ux1 - ux2;
    T_R const diffy = uy1 - uy2;
    T_R const diffz = uz1 - uz2;
    T_R const diffm = std::sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
    T_R const summ = std::sqrt(ux1*ux1 + uy1*uy1 + uz1*uz1) + std::sqrt(ux2*ux2 + uy2*uy2 + uz2*uz2);
    // If g = u1 - u2 = 0, do not collide.
    // Or if the relative difference is less than 1.0e-10.
    if (summ <= T_R(0.0)) return;
    if (diffm / summ < rel_tol) return;

    // ------------------------------------------------------------------
    // Initial lab-frame momenta
    // ------------------------------------------------------------------
    T_R const px1 = ux1 * m1 * PhysConstSI::c;
    T_R const py1 = uy1 * m1 * PhysConstSI::c;
    T_R const pz1 = uz1 * m1 * PhysConstSI::c;

    T_R const px2 = ux2 * m2 * PhysConstSI::c;
    T_R const py2 = uy2 * m2 * PhysConstSI::c;
    T_R const pz2 = uz2 * m2 * PhysConstSI::c;

    // ------------------------------------------------------------------
    // COM velocity
    // ------------------------------------------------------------------
    T_R const mass_g = m1 * g1 + m2 * g2;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mass_g > T_R(0.0), "Mass is zero in ImpactIonization");

    T_R const vcx    = (px1+px2) / mass_g;
    T_R const vcy    = (py1+py2) / mass_g;
    T_R const vcz    = (pz1+pz2) / mass_g;
    T_R const vcms   = vcx*vcx + vcy*vcy + vcz*vcz;

    T_R beta2 = vcms * inv_c2;
    if (beta2 < T_R(0.0)) {
        if (beta2 > -beta_tol) {
            beta2 = T_R(0.0);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                false,
                "ImpactIonization: beta < 0!"
            );
            return;
        }
    }

    if (beta2 >= T_R(1.0) - beta_tol) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            false,
            "ImpactIonization: beta too close to 1!"
        );
        return;
    }

    T_R const gc = T_R(1.0) / std::sqrt( T_R(1.0) - beta2 );

    // Stable equivalent of (gc - 1)/vcms:
    // (gamma - 1)/v^2 = gamma^2 / ((gamma + 1) c^2)
    T_R const boost_v2 = gc*gc / ((gc + T_R(1.0)) * c2);

    // ------------------------------------------------------------------
    // COM-frame boost of the initial momenta
    // ------------------------------------------------------------------
    T_R const vcDv1 = (vcx*ux1 + vcy*uy1 + vcz*uz1) * PhysConstSI::c / g1;
    T_R const vcDv2 = (vcx*ux2 + vcy*uy2 + vcz*uz2) * PhysConstSI::c / g2;

    // Boost particle 1 to COM
    T_R p1sx;
    T_R p1sy;
    T_R p1sz;
    if ( vcms > T_R(0.0) )
    {
        T_R const lorentz_coeff_1 =
            ( boost_v2 * vcDv1 - gc ) * m1 * g1;
        
        p1sx = px1 + vcx*lorentz_coeff_1;
        p1sy = py1 + vcy*lorentz_coeff_1;
        p1sz = pz1 + vcz*lorentz_coeff_1;
    }
    else // If vcms = 0, don't do Lorentz-transform.
    {
        p1sx = px1;
        p1sy = py1;
        p1sz = pz1;
    }

    // Boost particle 2 to COM
    T_R p2sx;
    T_R p2sy;
    T_R p2sz;
    if (vcms > T_R(0.0)) {
        T_R const lorentz_coeff_2 =
            ( boost_v2 * vcDv2 - gc ) * m2 * g2;
        
        p2sx = px2 + vcx*lorentz_coeff_2;
        p2sy = py2 + vcy*lorentz_coeff_2;
        p2sz = pz2 + vcz*lorentz_coeff_2;
    } else {
        p2sx = px2;
        p2sy = py2;
        p2sz = pz2;
    }

    // ------------------------------------------------------------------
    // Ionization probability and final-state construction
    // ------------------------------------------------------------------
    T_R const p1s_sq = p1sx*p1sx + p1sy*p1sy + p1sz*p1sz;
    T_R const p2s_sq = p2sx*p2sx + p2sy*p2sy + p2sz*p2sz;

    T_R const E1s = std::sqrt(p1s_sq*c2 + m1*m1*c2*c2);
    T_R const E2s = std::sqrt(p2s_sq*c2 + m2*m2*c2*c2);

    T_R const K1s = E1s - m1*c2;
    T_R const K2s = E2s - m2*c2;

    // Available initial kinetic energy in the COM frame [J]
    T_R const Ecm = K1s + K2s;

    // Kinetic energy available to the 3 outgoing particles in COM [J]
    T_R const Eout = Ecm - Eion;
    if (Eout <= T_R(0.0)) {     // Energy not sufficient for ionization
        AMREX_DEVICE_PRINTF(
            "ImpactIonization: energy not sufficient for ionization, E = %e = %e (Ecm) - %e (Eion)\n",
            Eout, Ecm, Eion
        );
        return;
    }    // Energy not sufficient for ionization

    // ------------------------------------------------------------------
    // 3-body final state in the COM frame
    // Particle 1 : scattered projectile electron   (mass m1)
    // Particle 2 : ion                             (mass m2)
    // Particle 3 : secondary electron              (mass m3)
    //
    // Coherent with WarpX electron impact model:
    // - Particles 1 and 3 (both electrons) get independent isotropic
    //   random directions in the COM frame.
    // - Particle 2 (ion) direction is fully determined by momentum
    //   conservation: p2 = -(p1 + p3).
    // - Equal momentum magnitudes |p1| = |p3| = C is justified by the
    //   equal energy share assumption (coherent with WarpX), which holds
    //   relativistically if m1 == m3, i.e. both particles are electrons.
    // - C is found by bisection so that total relativistic KE = Eout.
    // ------------------------------------------------------------------

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(m1 - m3) < T_R(1.0e-10) * m1,
        "ImpactIonization: equal momentum assumption requires m1 == m3 (both electrons)");

    // Sample independent isotropic unit vectors for the two electrons
    T_R const cos_theta1 = T_R(2.0) * amrex::Random(engine) - T_R(1.0);
    T_R const sin_theta1 = std::sqrt(amrex::max(T_R(0.0), T_R(1.0) - cos_theta1*cos_theta1));
    T_R const phi1       = amrex::Random(engine) * T_R(2.0) * MathConst::pi;
    T_R const e1x        = sin_theta1 * std::cos(phi1);
    T_R const e1y        = sin_theta1 * std::sin(phi1);
    T_R const e1z        = cos_theta1;

    T_R const cos_theta3 = T_R(2.0) * amrex::Random(engine) - T_R(1.0);
    T_R const sin_theta3 = std::sqrt(amrex::max(T_R(0.0), T_R(1.0) - cos_theta3*cos_theta3));
    T_R const phi3       = amrex::Random(engine) * T_R(2.0) * MathConst::pi;
    T_R const e3x        = sin_theta3 * std::cos(phi3);
    T_R const e3y        = sin_theta3 * std::sin(phi3);
    T_R const e3z        = cos_theta3;

    // Ion momentum direction from momentum conservation: p2 = -(p1 + p3)
    // With |p1| = |p3| = C: the ion momentum vector is -C*(e1 + e3)
    T_R const d2x    = -(e1x + e3x);
    T_R const d2y    = -(e1y + e3y);
    T_R const d2z    = -(e1z + e3z);
    T_R const n2_raw = std::sqrt(d2x*d2x + d2y*d2y + d2z*d2z);

    if (n2_raw < eps_n) {
        AMREX_DEVICE_PRINTF(
            "ImpactIonization: degenerate ion direction, n2_raw=%e\n", n2_raw);
        return;
    }

    T_R const ion_dx = d2x / n2_raw;
    T_R const ion_dy = d2y / n2_raw;
    T_R const ion_dz = d2z / n2_raw;

    // Bisection for C such that K1(C) + K2(C) + K3(C) = Eout
    // |p1| = C, |p3| = C, |p2| = C * n2_raw
    auto k_rel = [=] AMREX_GPU_HOST_DEVICE (T_R Crel) -> T_R
    {
        T_R const K1 = std::sqrt(Crel*Crel*c2 + m1*m1*c2*c2) - m1*c2;
        T_R const K2 = std::sqrt(Crel*Crel*n2_raw*n2_raw*c2 + m2*m2*c2*c2) - m2*c2;
        T_R const K3 = std::sqrt(Crel*Crel*c2 + m3*m3*c2*c2) - m3*c2;
        return K1 + K2 + K3;
    };

    T_R C_lo = T_R(0.0);
    T_R C_hi = std::sqrt(T_R(2.0) * m1 * Eout);
    while (k_rel(C_hi) < Eout) {
        C_hi *= T_R(2.0);
        if (C_hi > T_R(1.0e6) * std::sqrt(T_R(2.0) * m1 * Eout)) {
            AMREX_DEVICE_PRINTF("ImpactIonization: failed to bracket relativistic C\n");
            return;
        }
    }
    for (int iter = 0; iter < 60; ++iter) {
        T_R const C_mid = T_R(0.5) * (C_lo + C_hi);
        if (k_rel(C_mid) < Eout) C_lo = C_mid;
        else C_hi = C_mid;
    }
    T_R const C = T_R(0.5) * (C_lo + C_hi);

    // Final-state COM momenta
    T_R const p1fsx = C * e1x;
    T_R const p1fsy = C * e1y;
    T_R const p1fsz = C * e1z;

    T_R const p2fsx = C * n2_raw * ion_dx;
    T_R const p2fsy = C * n2_raw * ion_dy;
    T_R const p2fsz = C * n2_raw * ion_dz;

    T_R const p3fsx = C * e3x;
    T_R const p3fsy = C * e3y;
    T_R const p3fsz = C * e3z;
    
    // Boost final-state momenta from COM back to lab
    T_R const g1f_s = std::sqrt(T_R(1.0) + (p1fsx*p1fsx + p1fsy*p1fsy + p1fsz*p1fsz)/(m1*m1*c2));
    T_R const g2f_s = std::sqrt(T_R(1.0) + (p2fsx*p2fsx + p2fsy*p2fsy + p2fsz*p2fsz)/(m2*m2*c2));
    T_R const g3f_s = std::sqrt(T_R(1.0) + (p3fsx*p3fsx + p3fsy*p3fsy + p3fsz*p3fsz)/(m3*m3*c2));

    T_R p1fx; T_R p1fy; T_R p1fz;
    T_R p2fx; T_R p2fy; T_R p2fz;
    T_R p3fx; T_R p3fy; T_R p3fz;

    if ( vcms > T_R(0.0) )
    {
        T_R const vcDp1fs = vcx*p1fsx + vcy*p1fsy + vcz*p1fsz;
        T_R const vcDp2fs = vcx*p2fsx + vcy*p2fsy + vcz*p2fsz;
        T_R const vcDp3fs = vcx*p3fsx + vcy*p3fsy + vcz*p3fsz;

        T_R const factor1 = boost_v2*vcDp1fs + m1*g1f_s*gc;
        T_R const factor2 = boost_v2*vcDp2fs + m2*g2f_s*gc;
        T_R const factor3 = boost_v2*vcDp3fs + m3*g3f_s*gc;

        p1fx = p1fsx + vcx*factor1;
        p1fy = p1fsy + vcy*factor1;
        p1fz = p1fsz + vcz*factor1;

        p2fx = p2fsx + vcx*factor2;
        p2fy = p2fsy + vcy*factor2;
        p2fz = p2fsz + vcz*factor2;

        p3fx = p3fsx + vcx*factor3;
        p3fy = p3fsy + vcy*factor3;
        p3fz = p3fsz + vcz*factor3;
    }
    else // If vcms = 0, don't do Lorentz-transform
    {
        p1fx = p1fsx; p1fy = p1fsy; p1fz = p1fsz;
        p2fx = p2fsx; p2fy = p2fsy; p2fz = p2fsz;
        p3fx = p3fsx; p3fy = p3fsy; p3fz = p3fsz;
    }

    // Convert lab-frame momenta back to normalized momenta
    // Rejection method according to the adaptation of eq (14) in the ionization model, 
    // from Perez et al., Phys.Plasmas.19.083104 (2012)
    // The weights are rescaled according to eq (22)-(23),
    // Higginson et al., Journal of Computational Physics 413 (2020)
    auto r = amrex::Random(engine);
    if ( w2r > r*amrex::max(w1r, w2r) )
    {
        ux1 = inv_c * p1fx / m1;
        uy1 = inv_c * p1fy / m1;
        uz1 = inv_c * p1fz / m1;

        bool var_nan =
            std::isnan(ux1) || std::isnan(uy1) || std::isnan(uz1);

        bool var_inf =
            std::isinf(ux1) || std::isinf(uy1) || std::isinf(uz1);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!var_nan, "ImpactIonization: NaN detected in final-state velocities");

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!var_inf, "ImpactIonization: Inf detected in final-state velocities");
    }
    
    ion_lev2 += 1;

    ux2 = inv_c * p2fx / m2;
    uy2 = inv_c * p2fy / m2;
    uz2 = inv_c * p2fz / m2;

    ux3 = inv_c * p3fx / m3;
    uy3 = inv_c * p3fy / m3;
    uz3 = inv_c * p3fz / m3;

    bool var_nan =
        std::isnan(ux1) || std::isnan(uy1) || std::isnan(uz1) ||
        std::isnan(ux2) || std::isnan(uy2) || std::isnan(uz2) ||
        std::isnan(ux3) || std::isnan(uy3) || std::isnan(uz3);

    bool var_inf =
        std::isinf(ux1) || std::isinf(uy1) || std::isinf(uz1) ||
        std::isinf(ux2) || std::isinf(uy2) || std::isinf(uz2) ||
        std::isinf(ux3) || std::isinf(uy3) || std::isinf(uz3);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!var_nan, "ImpactIonization: NaN detected in final-state velocities");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!var_inf, "ImpactIonization: Inf detected in final-state velocities");
}

#endif // HIPACE_IMPACT_IONIZATION_H_