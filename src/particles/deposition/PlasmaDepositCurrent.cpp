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
void DepositCurrent_middle (int depos_order_xy, bool use_laser, bool do_tiling, bool can_ionize, Args&&...args)
{
    if (!use_laser && !do_tiling && !can_ionize) {
        switch (depos_order_xy) {
            case 0: return doDepositionShapeN<0, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 1: return doDepositionShapeN<1, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 2: return doDepositionShapeN<2, false, false, false>(use_laser, do_tiling, can_ionize, args...);
            case 3: return doDepositionShapeN<3, false, false, false>(use_laser, do_tiling, can_ionize, args...);
        }
    } else {
        switch (depos_order_xy) {
            case 0: return doDepositionShapeN<0, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 1: return doDepositionShapeN<1, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 2: return doDepositionShapeN<2, true, true, true>(use_laser, do_tiling, can_ionize, args...);
            case 3: return doDepositionShapeN<3, true, true, true>(use_laser, do_tiling, can_ionize, args...);
        }
    }
    amrex::Abort("unknow depos_order_xy: " + std::to_string(depos_order_xy));
}


void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields, const Laser& laser,
                const int which_slice, const bool temp_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                const bool deposit_j_squared, amrex::Geometry const& gm, int const lev,
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
    const bool can_ionize = plasma.m_can_ionize;
    const bool explicit_solve = Hipace::GetInstance().m_explicit;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract the fields currents
        // Do not access the field if the kernel later does not deposit into it,
        // the field might not be allocated. Use 0 as dummy component instead
        amrex::FArrayBox& isl_fab = fields.getSlices(lev, which_slice)[pti];
        const std::string plasma_str = explicit_solve && which_slice != WhichSlice::RhoIons ?
                                       "_" + plasma.GetName() : "";
        const int  jx_cmp = deposit_jx_jy     ? Comps[which_slice]["jx" +plasma_str] : -1;
        const int  jy_cmp = deposit_jx_jy     ? Comps[which_slice]["jy" +plasma_str] : -1;
        const int  jz_cmp = deposit_jz        ? Comps[which_slice]["jz" +plasma_str] : -1;
        const int rho_cmp = deposit_rho       ? Comps[which_slice]["rho"+plasma_str] : -1;
        const int jxx_cmp = deposit_j_squared ? Comps[which_slice]["jxx"+plasma_str] : -1;
        const int jxy_cmp = deposit_j_squared ? Comps[which_slice]["jxy"+plasma_str] : -1;
        const int jyy_cmp = deposit_j_squared ? Comps[which_slice]["jyy"+plasma_str] : -1;

        amrex::Vector<amrex::FArrayBox>& tmp_dens = fields.getTmpDensities();

        // extract the laser Fields
        const bool use_laser = laser.m_use_laser;
        const amrex::MultiFab& a_mf = laser.getSlices(WhichLaserSlice::This);

        // Offset for converting positions to indexes
        const amrex::Real x_pos_offset = GetPosOffset(0, gm, isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, isl_fab.box());
        const amrex::Real z_pos_offset = GetPosOffset(2, gm, isl_fab.box());

        DepositCurrent_middle(Hipace::m_depos_order_xy, use_laser, Hipace::m_do_tiling, can_ionize,
                              pti, isl_fab, jx_cmp, jy_cmp, jz_cmp, rho_cmp, jxx_cmp, jxy_cmp,
                              jyy_cmp, a_mf, tmp_dens, dx, x_pos_offset,
                              y_pos_offset, z_pos_offset, q, temp_slice,
                              deposit_jx_jy, deposit_jz, deposit_rho, deposit_j_squared,
                              max_qsa_weighting_factor, bins, bin_size);
    }
}
