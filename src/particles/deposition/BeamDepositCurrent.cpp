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
                     amrex::Vector<amrex::Geometry> const& gm, int const lev ,const int islice,
                     const bool do_beam_jx_jy_deposition,
                     const bool do_beam_jz_deposition,
                     const bool do_beam_rho_deposition,
                     const int which_slice, int nghost)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");

    using namespace amrex::literals;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm[lev].CellSize();

    // beam deposits only up to its finest level
    if (beam.m_finest_level < lev) return;

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

    // we deposit to the beam currents, because the explicit solver
    // requires sometimes just the beam currents
    // Do not access the field if the kernel later does not deposit into it,
    // the field might not be allocated. Use -1 as dummy component instead
    const std::string beam_str = Hipace::GetInstance().m_explicit ? "_beam" : "";
    const int  jxb_cmp = do_beam_jx_jy_deposition ? Comps[which_slice]["jx" +beam_str] : -1;
    const int  jyb_cmp = do_beam_jx_jy_deposition ? Comps[which_slice]["jy" +beam_str] : -1;
    const int  jzb_cmp = do_beam_jz_deposition    ? Comps[which_slice]["jz" +beam_str] : -1;
    const int rhob_cmp = do_beam_rho_deposition   ? Comps[which_slice]["rho"+beam_str] : -1;

    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
    amrex::Real const y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

    // Whether the ghost slice has to be deposited
    const bool deposit_ghost = ((which_slice==WhichSlice::Next) && (islice == 0));
    if (deposit_ghost && !do_beam_jx_jy_deposition) return;

    // Ghost particles are indexed [beam.numParticles()-nghost, beam.numParticles()-1]
    int box_offset = beam.m_box_sorter.boxOffsetsPtr()[beam.m_ibox];
    if (deposit_ghost) box_offset = beam.numParticles()-nghost;

    PhysConst const phys_const = get_phys_const();

    // Extract particle properties
    const auto& soa = beam.GetStructOfArrays(); // For momenta and weights
    const auto pos_x = soa.GetRealData(BeamIdx::x).data() + box_offset;
    const auto pos_y = soa.GetRealData(BeamIdx::y).data() + box_offset;
    const auto  wp = soa.GetRealData(BeamIdx::w).data() + box_offset;
    const auto uxp = soa.GetRealData(BeamIdx::ux).data() + box_offset;
    const auto uyp = soa.GetRealData(BeamIdx::uy).data() + box_offset;
    const auto uzp = soa.GetRealData(BeamIdx::uz).data() + box_offset;
    const auto idp = soa.GetIntData(BeamIdx::id).data() + box_offset;

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    amrex::Real invvol = dxi * dyi * dzi;

    if (Hipace::m_normalized_units) {
        if (lev == 0) {
            invvol = 1._rt;
        } else {
            // re-scaling the weight in normalized units to get the same charge density on lev 1
            // Not necessary in SI units, there the weight is the actual charge and not the density
            amrex::Real const * AMREX_RESTRICT dx_lev0 = gm[0].CellSize();
            invvol = dx_lev0[0] * dx_lev0[1] * dx_lev0[2] * dxi * dyi * dzi;
        }
    }

    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
    const amrex::Real q = beam.m_charge;

    Array3<amrex::Real> const isl_arr = isl_fab.array();

    BeamBins::index_type const * const indices = beam.m_slice_bins.permutationPtr();
    BeamBins::index_type const * const offsets = beam.m_slice_bins.offsetsPtrCpu();
    BeamBins::index_type cell_start = 0;
    BeamBins::index_type cell_stop = 0;

    // The particles that are in slice islice are
    // given by the indices[cell_start:cell_stop]
    if (which_slice == WhichSlice::This || which_slice == WhichSlice::Salame) {
        cell_start = offsets[islice];
        cell_stop  = offsets[islice+1];
    } else {
        if (islice > 0) {
            cell_start = offsets[islice-1];
            cell_stop  = offsets[islice];
        } else {
            cell_start = 0;
            cell_stop  = nghost;
        }
    }
    int const num_particles = cell_stop-cell_start;

    // Loop over particles and deposit into jx_fab, jy_fab, and jz_fab
    amrex::ParallelFor(
        amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
        {Hipace::m_depos_order_xy},
        num_particles,
        [=] AMREX_GPU_DEVICE (long idx, auto depos_order) {
            constexpr int depos_order_xy = depos_order.value;

            // Particles in the same slice must be accessed through the bin sorter.
            // Ghost particles are simply contiguous in memory.
            const int ip = deposit_ghost ? cell_start+idx : indices[cell_start+idx];

            // Skip invalid particles and ghost particles not in the last slice
            if (idp[ip] < 0) return;
            // --- Get particle quantities
            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                         + uyp[ip]*uyp[ip]*clightsq
                                                         + uzp[ip]*uzp[ip]*clightsq);
            const amrex::Real wq = q*wp[ip]*invvol;

            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;
            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = wq*vx;
            const amrex::Real wqy = wq*vy;
            const amrex::Real wqz = wq*vz;

            // --- Compute shape factors
            // x direction
            const amrex::Real xmid = (pos_x[ip] - x_pos_offset)*dxi;
            // i_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order_xy + 1];
            const int i_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid);

            // y direction
            const amrex::Real ymid = (pos_y[ip] - y_pos_offset)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid);

            // Deposit current into jx, jy, jz, rho
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
                    if (rhob_cmp != -1) { // do_beam_rho_deposition
                        amrex::Gpu::Atomic::Add(
                            isl_arr.ptr(i_cell+ix, j_cell+iy, rhob_cmp),
                            sx_cell[ix]*sy_cell[iy]*wq);
                    }
                }
            }
        }
        );

}
