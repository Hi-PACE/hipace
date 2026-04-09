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

/* \brief The implicit, non-relativistic assumption has not been
 *        changed
 *        Momentum and energy division after the ionization event
 *        is done as follows:
 *        Three numbers are generated that satisfy the triangle inequality.
 *        These numbers will be scaled to give the momentum
 *        for each particle (satisfying the triangle inequality
 *        ensures that the momentum vectors can be arranged
 *        such that the net linear momentum is zero).
 *        Species 2 is definitely a neutral/ion, so we first choose its
 *        proportion to be n_2 = min(E2 / Ecoll, 0.5 * R), where
 *        R is a random number between 0 and 1.
 *        The other two numbers (n_1 & n_3) are then found
 *        by choosing a random point on the ellipse characterized
 *        by the semi-major and semi-minor axes
 *        a = (1 - n_2) / 2.0
 *        b = 0.5 * sqrt(1 - 2 * n_2)
 *        The numbers are found by randomly sampling an x value
 *        between -a and a, and finding the corresponding y value
 *        that falls on the ellipse: y^2 = b^2 - b^2/a^2 * x^2.
 *        Then n_1 = sqrt(y^2 + (x - n_2/2)^2) and
 *        n_3 = 1 - n_2 - n_1.
 *        Next, we need to find the number C, such that if
 *        p_i = C * n_i, we would have E_1 + E_2 + E_3 = E_out
 *        where E_i = p_i^2 / (2 m_i), i.e.,
 *        C^2 * \sum_i [n_i^2 / (2 m_i)] = E_out
 *        After C is determined the momentum vectors are arranged
 *        such that the net linear momentum is zero.
 *        To see if there are nan or inf updated velocities,
 *        compile with USE_ASSERTION=TRUE.
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

    // Helper lambda to clamp a value between a lower and upper bound
    auto clamp = [] AMREX_GPU_HOST_DEVICE (T_R x, T_R lo, T_R hi) -> T_R
    {
        return amrex::max(lo, amrex::min(hi, x));
    };
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
            AMREX_DEVICE_PRINTF(
                "ImpactIonization: beta < 0, beta2=%e vcms=%e\n",
                beta2, vcms
            );
            return;
        }
    }

    if (beta2 >= T_R(1.0) - beta_tol) {
        AMREX_DEVICE_PRINTF(
            "ImpactIonization: beta too close to 1, beta2=%e\n",
            beta2
        );
        return;
    }

    T_R const gc = T_R(1.0) / std::sqrt( T_R(1.0) - beta2 );

    // Stable equivalent of (gc - 1)/vcms:
    // (gamma - 1)/v^2 = gamma^2 / ((gamma + 1) c^2)
    T_R const boost_v2 = gc*gc / ((gc + T_R(1.0)) * c2);

    // ------------------------------------------------------------------
    // COM-frame incoming particle 1 momentum
    // ------------------------------------------------------------------
    T_R const vcDv1 = (vcx*ux1 + vcy*uy1 + vcz*uz1) * PhysConstSI::c / g1;
    T_R const vcDv2 = (vcx*ux2 + vcy*uy2 + vcz*uz2) * PhysConstSI::c / g2;

    // Compute p1 star
    T_R p1sx;
    T_R p1sy;
    T_R p1sz;
    if ( vcms > std::numeric_limits<T_R>::min() )
    {
        T_R const lorentz_transform_factor =
            ( boost_v2 * vcDv1 - gc ) * m1 * g1;
        
        p1sx = px1 + vcx*lorentz_transform_factor;
        p1sy = py1 + vcy*lorentz_transform_factor;
        p1sz = pz1 + vcz*lorentz_transform_factor;
    }
    else // If vcms = 0, don't do Lorentz-transform.
    {
        p1sx = px1;
        p1sy = py1;
        p1sz = pz1;
    }

    // ------------------------------------------------------------------
    // COM-frame gammas
    // ------------------------------------------------------------------
    T_R const g1s = ( T_R(1.0) - vcDv1*inv_c2 )*gc*g1;
    T_R const g2s = ( T_R(1.0) - vcDv2*inv_c2 )*gc*g2;
    // Redundant?
    if (g1s < T_R(1.0) || g2s < T_R(1.0)) {
        AMREX_DEVICE_PRINTF(
            "ImpactIonization: invalid COM gammas, g1s=%e g2s=%e\n",
            g1s, g2s
        );
        return;
    }

    // Ionization probability and final-state construction
    // Available initial kinetic energy in the COM frame [J]
    T_R const Ecm =
        ( (g1s - T_R(1.0))*m1 + (g2s - T_R(1.0))*m2 ) * c2;

    // Kinetic energy available to the 3 outgoing particles in COM [J]
    T_R const Eout = Ecm - Eion;
    if (Eout <= T_R(0.0)) return;   // Energy not sufficient for ionization

    // 3-body final state in the COM frame
    // Particle 1 : scattered projectile       (mass m1)
    // Particle 2 : ion / neutral              (mass m2)
    // Particle 3 : secondary electron         (mass m3)
    //
    // Momentum magnitudes satisfy triangle closure,
    // scaled so that total kinetic energy = Eout.

    T_R const E_target = (g2s-T_R(1.0))*m2 * c2;
    // Sample three normalized momentum magnitudes satisfying triangle inequality
    T_R n_2 = amrex::min(E_target/Ecm, T_R(0.5)*amrex::Random(engine));
    n_2 = clamp(n_2, T_R(0.0), T_R(0.5) - eps_n);

    T_R const a = T_R(0.5) * (T_R(1.0) - n_2);

    T_R const b2_arg = T_R(1.0) - T_R(2.0)*n_2;
    if (b2_arg < T_R(0.0)) {
        AMREX_DEVICE_PRINTF(
            "ImpactIonization: invalid ellipse, b2_arg=%e n2=%e E_target=%e Ecm=%e\n",
            b2_arg, n_2, E_target, Ecm
        );
        return;
    }

    T_R const b = T_R(0.5) * std::sqrt(amrex::max(T_R(0.0), b2_arg));

    T_R const x  = (T_R(2.0)*amrex::Random(engine) - T_R(1.0)) * a;
    T_R y2 = b*b - (b*b/(a*a))*x*x;
    y2 = amrex::max(T_R(0.0), y2);

    T_R n1sq = y2 + x*x - x*n_2 + T_R(0.25)*n_2*n_2;
    n1sq = amrex::max(T_R(0.0), n1sq);

    T_R const n_1 = std::sqrt(n1sq);
    T_R const n_3 = T_R(1.0) - n_1 - n_2;

    T_R const den_alpha = T_R(2.0)*n_1*n_3;
    T_R const den_gamma = T_R(2.0)*n_1*n_2;

    if (n_1 <= eps_n || n_2 <= eps_n || n_3 <= eps_n || den_alpha <= eps_n || den_gamma <= eps_n) {
        AMREX_DEVICE_PRINTF("ImpactIonization: degenerate triangle n1=%e n2=%e n3=%e\n",
            n_1, n_2, n_3);
        return;
    }

    T_R const C_den = n_1*n_1 / (T_R(2.0)*m1) + n_2*n_2 / (T_R(2.0)*m2) + n_3*n_3 / (T_R(2.0)*m3);
    if (C_den <= eps_n) {
        AMREX_DEVICE_PRINTF("ImpactIonization: bad energy_den=%e\n", C_den);
        return;
    }
    // Common momentum scale from energy conservation (non-relativistic!)
    T_R const C = std::sqrt(Eout / C_den);

    // Momentum magnitudes in COM
    T_R const p1f_mag = C * n_1;
    T_R const p2f_mag = C * n_2;
    T_R const p3f_mag = C * n_3;

    // Triangle angles from the law of cosines
    T_R cos_alpha = (n_1*n_1 + n_3*n_3 - n_2*n_2) / den_alpha;
    cos_alpha = clamp(cos_alpha, T_R(-1.0), T_R(1.0));
    
    T_R cos_gamma = (n_1*n_1 + n_2*n_2 - n_3*n_3) / den_gamma;
    cos_gamma = clamp(cos_gamma, T_R(-1.0), T_R(1.0));

    T_R const sin_alpha = std::sqrt(amrex::max(T_R(0.0), T_R(1.0) - cos_alpha*cos_alpha));
    T_R const sin_gamma = std::sqrt(amrex::max(T_R(0.0), T_R(1.0) - cos_gamma*cos_gamma));

    // Random orientation of the momentum triangle in 3D
    T_R const Theta = amrex::Random(engine) * T_R(2.0) * MathConst::pi;
    T_R const phi   = amrex::Random(engine) * MathConst::pi;

    T_R const cos_theta = std::cos(Theta);
    T_R const sin_theta = std::sin(Theta);
    T_R const cos_phi   = std::cos(phi);
    T_R const sin_phi   = std::sin(phi);

    // Outgoing momenta in the COM frame
    T_R p1fsx = p1f_mag * cos_theta * cos_phi;
    T_R p1fsy = p1f_mag * cos_theta * sin_phi;
    T_R p1fsz = -p1f_mag * sin_theta;

    T_R p2fsx = p2f_mag * (-cos_alpha*cos_theta*cos_phi - sin_alpha*sin_phi);
    T_R p2fsy = p2f_mag * (-cos_alpha*cos_theta*sin_phi + sin_alpha*cos_phi);
    T_R p2fsz = p2f_mag * ( cos_alpha*sin_theta );

    T_R p3fsx = p3f_mag * (-cos_gamma*cos_theta*cos_phi + sin_gamma*sin_phi);
    T_R p3fsy = p3f_mag * (-cos_gamma*cos_theta*sin_phi - sin_gamma*cos_phi);
    T_R p3fsz = p3f_mag * ( cos_gamma*sin_theta );

    // Boost final-state momenta from COM back to lab
    T_R const g1f_s = std::sqrt(T_R(1.0) + (p1fsx*p1fsx + p1fsy*p1fsy + p1fsz*p1fsz)/(m1*m1*c2));
    T_R const g2f_s = std::sqrt(T_R(1.0) + (p2fsx*p2fsx + p2fsy*p2fsy + p2fsz*p2fsz)/(m2*m2*c2));
    T_R const g3f_s = std::sqrt(T_R(1.0) + (p3fsx*p3fsx + p3fsy*p3fsy + p3fsz*p3fsz)/(m3*m3*c2));

    T_R p1fx; T_R p1fy; T_R p1fz;
    T_R p2fx; T_R p2fy; T_R p2fz;
    T_R p3fx; T_R p3fy; T_R p3fz;

    if ( vcms > std::numeric_limits<T_R>::min() )
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