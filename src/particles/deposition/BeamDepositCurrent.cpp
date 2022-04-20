/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Remi Lehe
 * Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "BeamDepositCurrent.H"
#include "particles/BeamParticleContainer.H"
#include "particles/deposition/BeamDepositCurrentInner.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_DenseBins.H>

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Vector<amrex::Geometry> const& gm, int const lev ,const int islice,
                     int const offset, const BeamBins& bins, const bool do_beam_jx_jy_deposition,
                     const int which_slice, int nghost, const bool do_beam_jz_minus_rho)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm[lev].CellSize();

    // beam deposits only up to its finest level
    if (beam.m_finest_level < lev) return;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next,
    "Current deposition can only be done in this slice (WhichSlice::This), or the next slice "
    " (WhichSlice::Next)");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0,
        "Only order 0 deposition is allowed for beam per-slice deposition");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(which_slice != WhichSlice::This && do_beam_jz_minus_rho),
        "depositing jz and rho beam in a slice that is not This");

    // Extract the fields currents
    amrex::MultiFab& S = fields.getSlices(lev, which_slice);
    // we deposit to the beam currents, because the explicit solver
    // requires sometimes just the beam currents
    amrex::MultiFab jx_beam(S, amrex::make_alias, Comps[which_slice]["jx_beam"], 1);
    amrex::MultiFab jy_beam(S, amrex::make_alias, Comps[which_slice]["jy_beam"], 1);
    amrex::MultiFab jz_beam(S, amrex::make_alias, Comps[which_slice]["jz_beam"], 1);
    amrex::MultiFab rho_beam(S, amrex::make_alias, Comps[which_slice]["rho_beam"], 1);

    // Extract FabArray for this box (because there is currently no transverse
    // parallelization, the index we want in the slice multifab is always 0.
    // Fix later.
    amrex::FArrayBox& jxb_fab = jx_beam[0];
    amrex::FArrayBox& jyb_fab = jy_beam[0];
    amrex::FArrayBox& jzb_fab = jz_beam[0];
    amrex::FArrayBox& rhob_fab = rho_beam[0];

    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset = GetPosOffset(0, gm[lev], jxb_fab.box());
    const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], jxb_fab.box());
    amrex::Real const z_pos_offset = GetPosOffset(2, gm[lev], jxb_fab.box());

    amrex::Real lev_weight_fac = 1.;
    if (lev == 1 && Hipace::m_normalized_units) {
        // re-scaling the weight in normalized units to get the same charge density on lev 1
        // Not necessary in SI units, there the weight is the actual charge and not the density
        amrex::Real const * AMREX_RESTRICT dx_lev0 = gm[0].CellSize();
        lev_weight_fac = dx_lev0[0] * dx_lev0[1] * dx_lev0[2] / (dx[0] * dx[1] * dx[2]);
    }

    // For now: fix the value of the charge
    const amrex::Real q = beam.m_charge * lev_weight_fac;

    // Call deposition function in each box
    if        (Hipace::m_depos_order_xy == 0){
        doDepositionShapeN<0, 0>( beam, jxb_fab, jyb_fab, jzb_fab, rhob_fab, dx, x_pos_offset, y_pos_offset,
                                  z_pos_offset, q, islice, bins, offset, do_beam_jx_jy_deposition,
                                  which_slice, nghost, do_beam_jz_minus_rho);
    } else if (Hipace::m_depos_order_xy == 1){
        doDepositionShapeN<1, 0>( beam, jxb_fab, jyb_fab, jzb_fab, rhob_fab, dx, x_pos_offset, y_pos_offset,
                                  z_pos_offset, q, islice, bins, offset, do_beam_jx_jy_deposition,
                                  which_slice, nghost, do_beam_jz_minus_rho);
    } else if (Hipace::m_depos_order_xy == 2){
        doDepositionShapeN<2, 0>( beam, jxb_fab, jyb_fab, jzb_fab, rhob_fab, dx, x_pos_offset, y_pos_offset,
                                  z_pos_offset, q, islice, bins, offset, do_beam_jx_jy_deposition,
                                  which_slice, nghost, do_beam_jz_minus_rho);
    } else if (Hipace::m_depos_order_xy == 3){
        doDepositionShapeN<3, 0>( beam, jxb_fab, jyb_fab, jzb_fab, rhob_fab, dx, x_pos_offset, y_pos_offset,
                                  z_pos_offset, q, islice, bins, offset, do_beam_jx_jy_deposition,
                                  which_slice, nghost, do_beam_jz_minus_rho);
    } else {
        amrex::Abort("unknown deposition order");
    }

}
