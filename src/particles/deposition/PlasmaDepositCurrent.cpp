/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaDepositCurrent.H"

#include "particles/PlasmaParticleContainer.H"
#include "particles/deposition/PlasmaDepositCurrentInner.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

template<class...Args>
void DepositCurrent_middle (bool outer_depos_loop, int depos_order_xy, bool use_laser,
                            bool do_tiling, bool can_ionize, Args&&...args)
{
    if (outer_depos_loop && !use_laser && !do_tiling && !can_ionize) {
        switch (depos_order_xy) {
            case 0: return doDepositionShapeN<true, 0, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 1: return doDepositionShapeN<true, 1, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 2: return doDepositionShapeN<true, 2, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 3: return doDepositionShapeN<true, 3, false, false, false>(use_laser, do_tiling, can_ionize, args...);
        }
    } else if (!outer_depos_loop && !use_laser && !do_tiling && !can_ionize) {
        switch (depos_order_xy) {
            case 0: return doDepositionShapeN<false, 0, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 1: return doDepositionShapeN<false, 1, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 2: return doDepositionShapeN<false, 2, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 3: return doDepositionShapeN<false, 3, false, false, false>(use_laser, do_tiling, can_ionize, args...);
        }
    } else if (outer_depos_loop) {
        switch (depos_order_xy) {
            case 0: return doDepositionShapeN<true, 0, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 1: return doDepositionShapeN<true, 1, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 2: return doDepositionShapeN<true, 2, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 3: return doDepositionShapeN<true, 3, true, true, true>(use_laser, do_tiling, can_ionize, args...);
        }
    } else {
        switch (depos_order_xy) {
            case 0: return doDepositionShapeN<false, 0, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 1: return doDepositionShapeN<false, 1, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 2: return doDepositionShapeN<false, 2, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 3: return doDepositionShapeN<false, 3, true, true, true>(use_laser, do_tiling, can_ionize, args...);
        }
    }
    amrex::Abort("unknow depos_order_xy: " + std::to_string(depos_order_xy));
}


void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields, const Laser& laser,
                const int which_slice, const bool temp_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                const bool deposit_chi, amrex::Geometry const& gm, int const lev,
                const PlasmaBins& bins, int bin_size)
{
    HIPACE_PROFILE("DepositCurrent_PlasmaParticleContainer()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next ||
    which_slice == WhichSlice::RhoIons,
    "Current deposition can only be done in this slice (WhichSlice::This), the next slice "
    " (WhichSlice::Next) or for the ion charge deposition (WhichSLice::RhoIons)");

    // only deposit plasma currents on their according MR level
    if (plasma.m_level != lev) return;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    const amrex::Real max_qsa_weighting_factor = plasma.m_max_qsa_weighting_factor;
    const amrex::Real q = (which_slice == WhichSlice::RhoIons) ? -plasma.m_charge : plasma.m_charge;
    const amrex::Real mass = plasma.m_mass;
    const bool can_ionize = plasma.m_can_ionize;
    const bool explicit_solve = Hipace::GetInstance().m_explicit;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract the fields currents
        // Do not access the field if the kernel later does not deposit into it,
        // the field might not be allocated. Use 0 as dummy component instead
        amrex::FArrayBox& isl_fab = fields.getSlices(lev, which_slice)[pti];
        const int  jx_cmp = deposit_jx_jy ? Comps[which_slice]["jx"]  : -1;
        const int  jy_cmp = deposit_jx_jy ? Comps[which_slice]["jy"]  : -1;
        const int  jz_cmp = deposit_jz    ? Comps[which_slice]["jz"]  : -1;
        const int rho_cmp = deposit_rho   ? Comps[which_slice]["rho"] : -1;
        const int chi_cmp = deposit_chi   ? Comps[which_slice]["chi"] : -1;

        amrex::Vector<amrex::FArrayBox>& tmp_dens = fields.getTmpDensities();

        // extract the laser Fields
        const bool use_laser = laser.m_use_laser;
        const amrex::MultiFab& a_mf = laser.getSlices(WhichLaserSlice::n00j00);

        // Offset for converting positions to indexes
        const amrex::Real x_pos_offset = GetPosOffset(0, gm, isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, isl_fab.box());
        const amrex::Real z_pos_offset = GetPosOffset(2, gm, isl_fab.box());

        DepositCurrent_middle(Hipace::m_outer_depos_loop, Hipace::m_depos_order_xy,
                              use_laser, Hipace::m_do_tiling, can_ionize,
                              pti, isl_fab, jx_cmp, jy_cmp, jz_cmp, rho_cmp, chi_cmp,
                              a_mf, tmp_dens, dx, x_pos_offset,
                              y_pos_offset, z_pos_offset, q, mass, temp_slice,
                              max_qsa_weighting_factor, bins, bin_size);
    }
}
