/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef BEAMPARTICLEADVANCE_H_
#define BEAMPARTICLEADVANCE_H_

#include "particles/beam/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "Hipace.H"

template <int depos_order>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void InterpolateEzInZ (
    amrex::Real& Ezp,
    const amrex::Real xp, const amrex::Real yp, const amrex::Real zp,
    const amrex::Real x_pos_offset, const amrex::Real y_pos_offset, const amrex::Real min_z,
    const amrex::Real dx_inv, const amrex::Real dy_inv, const amrex::Real dz_inv,
    const int ez_comp_prev, const int ez_comp_next, const Array3<const amrex::Real> slice_arr
)
{
    using namespace amrex::literals;

    // x,y,z direction
    const amrex::Real xmid = (xp-x_pos_offset)*dx_inv;
    const amrex::Real ymid = (yp-y_pos_offset)*dy_inv;
    const amrex::Real zmid = (zp-min_z)*dz_inv-0.5_rt;

    const auto [shape_p, pcell] = shape_factor<2>(zmid, 2);
    const auto [shape_n, ncell] = shape_factor<2>(zmid, 0);

    Ezp *= (1._rt - shape_p - shape_n);

    // Gather Ez field on particle from grid
    for (int iy=0; iy<=depos_order; iy++){
        for (int ix=0; ix<=depos_order; ix++){
            // Compute shape factors
            auto [shape_y, jcell] = shape_factor<depos_order>(ymid, iy);
            auto [shape_x, icell] = shape_factor<depos_order>(xmid, ix);

            Ezp += shape_p * shape_y * shape_x * slice_arr(icell, jcell, ez_comp_prev);
            Ezp += shape_n * shape_y * shape_x * slice_arr(icell, jcell, ez_comp_next);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void PushSpin (
    amrex::RealVect& spin,
    const amrex::Real ExmByp, const amrex::Real EypBxp, const amrex::Real Ezp,
    const amrex::Real Bxp, const amrex::Real Byp, const amrex::Real Bzp,
    const amrex::Real ux_intermediate, const amrex::Real uy_intermediate,
    const amrex::Real uz_intermediate, const amrex::Real gamma_intermediate_inv,
    const amrex::Real charge_mass_ratio, const amrex::Real dt,
    const amrex::Real spin_anom
)
{
    using namespace amrex::literals;

    const amrex::RealVect E {ExmByp + Byp, EypBxp - Bxp, Ezp};
    const amrex::RealVect B {Bxp, Byp, Bzp};
    const amrex::RealVect u {ux_intermediate, uy_intermediate, uz_intermediate};
    const amrex::RealVect beta = u*gamma_intermediate_inv;
    const amrex::Real gamma_inv_p1 =
        gamma_intermediate_inv / (1._rt + gamma_intermediate_inv);

    const amrex::RealVect omega = std::abs(charge_mass_ratio) * (
        B * gamma_intermediate_inv - beta.crossProduct(E) * gamma_inv_p1
        + spin_anom * (
            B - gamma_inv_p1 * u * beta.dotProduct(B) - beta.crossProduct(E)
        )
    );

    const amrex::RealVect h = omega * dt * 0.5_rt;
    const amrex::RealVect s_prime = spin + h.crossProduct(spin);
    const amrex::Real o = 1._rt / (1._rt + h.dotProduct(h));
    spin = o * (s_prime + (h.dotProduct(s_prime) * h + h.crossProduct(s_prime)));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ApplyRadiationReaction (
    amrex::Real& ux_next, amrex::Real& uy_next, amrex::Real& uz_next,
    const amrex::Real ExmByp, const amrex::Real EypBxp, const amrex::Real Ezp,
    const amrex::Real Bxp, const amrex::Real Byp, const amrex::Real Bzp,
    const amrex::Real ux_intermediate, const amrex::Real uy_intermediate,
    const amrex::Real uz_intermediate, const amrex::Real gamma_intermediate_inv,
    const amrex::Real dt, const amrex::Real rr_factor
)
{
    using namespace amrex::literals;

    const amrex::Real Exp = ExmByp + Byp;
    const amrex::Real Eyp = EypBxp - Bxp;

    const amrex::Real gamma_intermediate = std::sqrt( 1._rt
        + ux_intermediate*ux_intermediate
        + uy_intermediate*uy_intermediate
        + uz_intermediate*uz_intermediate);

    // Estimation of normalized velocity beta (v/c) at intermediate time
    const amrex::Real bx_n = ux_intermediate * gamma_intermediate_inv;
    const amrex::Real by_n = uy_intermediate * gamma_intermediate_inv;
    const amrex::Real bz_n = uz_intermediate * gamma_intermediate_inv;

    // Lorentz force over charge
    const amrex::Real flx_q = (Exp + by_n*Bzp - bz_n*Byp);
    const amrex::Real fly_q = (Eyp + bz_n*Bxp - bx_n*Bzp);
    const amrex::Real flz_q = (Ezp + bx_n*Byp - by_n*Bxp);
    const amrex::Real fl_q2 = flx_q*flx_q + fly_q*fly_q + flz_q*flz_q;

    // Calculation of auxiliary quantities
    const amrex::Real bdotE = (bx_n*Exp + by_n*Eyp + bz_n*Ezp);
    const amrex::Real bdotE2 = bdotE*bdotE;
    const amrex::Real coeff = gamma_intermediate*gamma_intermediate*(fl_q2-bdotE2);

    // Compute the components of the RR force
    const amrex::Real frx = fly_q*Bzp - flz_q*Byp + bdotE*Exp - coeff*bx_n;
    const amrex::Real fry = flz_q*Bxp - flx_q*Bzp + bdotE*Eyp - coeff*by_n;
    const amrex::Real frz = flx_q*Byp - fly_q*Bxp + bdotE*Ezp - coeff*bz_n;

    // Update momentum using the RR force
    ux_next += dt * rr_factor * frx;
    uy_next += dt * rr_factor * fry;
    uz_next += dt * rr_factor * frz;
}

struct MRLevelData {
    // Array to access fields
    Array3<const amrex::Real> slice_arr;
    // Properties associated with physical size of the box
    amrex::Real dx_inv = 0;
    amrex::Real dy_inv = 0;
    // Offset for converting positions to indexes
    amrex::Real x_pos_offset = 0;
    amrex::Real y_pos_offset = 0;
};

/** Push beam particles contained in one z slice
 * \param[in,out] beam species of which the current is deposited
 * \param[in] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] slice longitudinal slice
 * \param[in] current_N_level number of MR levels active on the current slice
 */
void
AdvanceBeamParticlesSlice (
    BeamParticleContainer& beam, const Fields& fields, amrex::Vector<amrex::Geometry> const& gm,
    const int slice, int const current_N_level);

#endif //  BEAMPARTICLEADVANCE_H_
