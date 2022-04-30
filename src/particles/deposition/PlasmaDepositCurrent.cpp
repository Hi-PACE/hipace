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

void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields, const Laser& laser,
                const int which_slice, const bool temp_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                bool deposit_j_squared, amrex::Geometry const& gm, int const lev,
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
        amrex::MultiFab& S = fields.getSlices(lev, which_slice);
        const std::string plasma_str = explicit_solve && which_slice != WhichSlice::RhoIons ? "_" + plasma.m_name : "";
        amrex::MultiFab jx(S,  amrex::make_alias, deposit_jx_jy     ? Comps[which_slice]["jx" +plasma_str] : 0, 1);
        amrex::MultiFab jy(S,  amrex::make_alias, deposit_jx_jy     ? Comps[which_slice]["jy" +plasma_str] : 0, 1);
        amrex::MultiFab jz(S,  amrex::make_alias, deposit_jz        ? Comps[which_slice]["jz" +plasma_str] : 0, 1);
        amrex::MultiFab rho(S, amrex::make_alias, deposit_rho       ? Comps[which_slice]["rho"+plasma_str] : 0, 1);
        amrex::MultiFab jxx(S, amrex::make_alias, deposit_j_squared ? Comps[which_slice]["jxx"+plasma_str] : 0, 1);
        amrex::MultiFab jxy(S, amrex::make_alias, deposit_j_squared ? Comps[which_slice]["jxy"+plasma_str] : 0, 1);
        amrex::MultiFab jyy(S, amrex::make_alias, deposit_j_squared ? Comps[which_slice]["jyy"+plasma_str] : 0, 1);
        amrex::Vector<amrex::FArrayBox>& tmp_dens = fields.getTmpDensities();

        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jz_fab = jz[pti];
        amrex::FArrayBox& rho_fab = rho[pti];
        amrex::FArrayBox& jxx_fab = jxx[pti];
        amrex::FArrayBox& jxy_fab = jxy[pti];
        amrex::FArrayBox& jyy_fab = jyy[pti];

        // extract the laser Fields
        const bool use_laser = laser.m_use_laser;
        const amrex::MultiFab& a_mf = laser.getSlices(WhichLaserSlice::This);

        // Offset for converting positions to indexes
        const amrex::Real x_pos_offset = GetPosOffset(0, gm, jx_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, jx_fab.box());
        const amrex::Real z_pos_offset = GetPosOffset(2, gm, jx_fab.box());


        if        (Hipace::m_depos_order_xy == 0){
                doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, jxx_fab, jxy_fab,
                                          jyy_fab, a_mf, use_laser, tmp_dens, dx, x_pos_offset,
                                          y_pos_offset, z_pos_offset, q, can_ionize, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho, deposit_j_squared,
                                          max_qsa_weighting_factor, bins, bin_size);
        } else if (Hipace::m_depos_order_xy == 1){
                doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, jxx_fab, jxy_fab,
                                          jyy_fab, a_mf, use_laser, tmp_dens, dx, x_pos_offset,
                                          y_pos_offset, z_pos_offset, q, can_ionize, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho, deposit_j_squared,
                                          max_qsa_weighting_factor, bins, bin_size);
        } else if (Hipace::m_depos_order_xy == 2){
                doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, jxx_fab, jxy_fab,
                                          jyy_fab, a_mf, use_laser, tmp_dens, dx, x_pos_offset,
                                          y_pos_offset, z_pos_offset, q, can_ionize, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho, deposit_j_squared,
                                          max_qsa_weighting_factor, bins, bin_size);
        } else if (Hipace::m_depos_order_xy == 3){
                doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, jxx_fab, jxy_fab,
                                          jyy_fab, a_mf, use_laser, tmp_dens, dx, x_pos_offset,
                                          y_pos_offset, z_pos_offset, q, can_ionize, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho, deposit_j_squared,
                                          max_qsa_weighting_factor, bins, bin_size);
        } else {
            amrex::Abort("unknow deposition order");
        }
    }
}
