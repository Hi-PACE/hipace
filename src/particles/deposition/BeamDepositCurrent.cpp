/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Remi Lehe
 * Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "BeamDepositCurrent.H"
#include "particles/beam/BeamParticleContainer.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

#include <AMReX_DenseBins.H>

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Vector<amrex::Geometry> const& gm, int const lev,
                     const bool do_beam_jx_jy_deposition,
                     const bool do_beam_jz_deposition,
                     const bool do_beam_rhomjz_deposition,
                     const int which_slice, const int which_beam_slice, const bool only_highest)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");

    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next
    || which_slice == WhichSlice::Salame,
    "Current deposition can only be done in this slice (WhichSlice::This), the next slice "
    " (WhichSlice::Next), or the SALAME slice (WhichSlice::Salame)");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0,
        "Only order 0 deposition is allowed for beam per-slice deposition");

    // Extract the fields currents
    // Extract FabArray for this box (because there is currently no transverse
    // parallelization, the index we want in the slice multifab is always 0.
    // Fix later.
    amrex::FArrayBox& isl_fab = fields.getSlices(lev)[0];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(isl_fab.box().ixType().cellCentered(),
        "jx, jy, jz and rho must be nodal in all directions.");

    if (!do_beam_jx_jy_deposition && !do_beam_jz_deposition && !do_beam_rhomjz_deposition) return;

    // we deposit to the beam currents, because the explicit solver
    // requires sometimes just the beam currents
    // Do not access the field if the kernel later does not deposit into it,
    // the field might not be allocated. Use -1 as dummy component instead
    const std::string beam_str = Hipace::m_explicit ? "_beam" : "";
    const int     jxb_cmp = do_beam_jx_jy_deposition  ? Comps[which_slice]["jx"    +beam_str] : -1;
    const int     jyb_cmp = do_beam_jx_jy_deposition  ? Comps[which_slice]["jy"    +beam_str] : -1;
    const int     jzb_cmp = do_beam_jz_deposition     ? Comps[which_slice]["jz"    +beam_str] : -1;
    const int rhomjzb_cmp = do_beam_rhomjz_deposition ? Comps[which_slice]["rhomjz"+beam_str] : -1;

    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
    amrex::Real const y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

    PhysConst const phys_const = get_phys_const();

    // Extract particle properties
    const auto ptd = beam.getBeamSlice(which_beam_slice).getParticleTileData();

    // Extract box properties
    const amrex::Real dxi = gm[lev].InvCellSize(0);
    const amrex::Real dyi = gm[lev].InvCellSize(1);
    const amrex::Real dzi = gm[lev].InvCellSize(2);
    amrex::Real invvol = dxi * dyi * dzi;

    if (Hipace::m_normalized_units) {
        if (lev == 0) {
            invvol = 1._rt;
        } else {
            // re-scaling the weight in normalized units to get the same charge density as on lev 0
            // Not necessary in SI units, there the weight is the actual charge and not the density
            invvol = gm[0].CellSize(0) * gm[0].CellSize(1) * dxi * dyi;
        }
    }

    const amrex::Real clightinv = 1.0_rt/(phys_const.c);
    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
    const amrex::Real q = beam.m_charge;

    Array3<amrex::Real> const isl_arr = isl_fab.array();

    // Loop over particles and deposit into jx_fab, jy_fab, and jz_fab
    amrex::ParallelFor(
        amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
        {Hipace::m_depos_order_xy},
        beam.getNumParticles(which_beam_slice),
        [=] AMREX_GPU_DEVICE (long ip, auto depos_order) {
            constexpr int depos_order_xy = depos_order.value;

            // Skip invalid particles and ghost particles not in the last slice
            // beam deposits only up to its finest level
            if (!ptd.id(ip).is_valid() ||
                (only_highest ?
                    (ptd.idata(BeamIdx::mr_level)[ip] != lev) :
                    (ptd.idata(BeamIdx::mr_level)[ip] < lev))
            ) return;

            // --- Get particle quantities
            const amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            const amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            const amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + ux*ux*clightsq
                                                         + uy*uy*clightsq
                                                         + uz*uz*clightsq);
            const amrex::Real wq = q*ptd.rdata(BeamIdx::w)[ip]*invvol;

            const amrex::Real vx = ux*gaminv;
            const amrex::Real vy = uy*gaminv;
            const amrex::Real vz = uz*gaminv;
            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = wq*vx;
            const amrex::Real wqy = wq*vy;
            const amrex::Real wqz = wq*vz;
            const amrex::Real wqrhomjz = wq*(1._rt-vz*clightinv);

            // --- Compute shape factors
            // x direction
            const amrex::Real xmid = (ptd.pos(0, ip) - x_pos_offset)*dxi;
            // i_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order_xy + 1];
            const int i_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid);

            // y direction
            const amrex::Real ymid = (ptd.pos(1, ip) - y_pos_offset)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid);

            // Deposit current into jx, jy, jz, rhomjz
            for (int iy=0; iy<=depos_order_xy; iy++){
                for (int ix=0; ix<=depos_order_xy; ix++){
                    if (jxb_cmp != -1) { // do_beam_jx_jy_deposition
                        amrex::Gpu::Atomic::Add(
                            isl_arr.ptr(i_cell+ix, j_cell+iy, jxb_cmp),
                            sx_cell[ix]*sy_cell[iy]*wqx);
                        amrex::Gpu::Atomic::Add(
                            isl_arr.ptr(i_cell+ix, j_cell+iy, jyb_cmp),
                            sx_cell[ix]*sy_cell[iy]*wqy);
                    }
                    if (jzb_cmp != -1) { // do_beam_jz_deposition
                        amrex::Gpu::Atomic::Add(
                            isl_arr.ptr(i_cell+ix, j_cell+iy, jzb_cmp),
                            sx_cell[ix]*sy_cell[iy]*wqz);
                    }
                    if (rhomjzb_cmp != -1) { // do_beam_rhomjz_deposition
                        amrex::Gpu::Atomic::Add(
                            isl_arr.ptr(i_cell+ix, j_cell+iy, rhomjzb_cmp),
                            sx_cell[ix]*sy_cell[iy]*wqrhomjz);
                    }
                }
            }
        }
        );

}
