/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Remi Lehe
 * Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_BEAMDEPOSITCURRENTINNER_H_
#define HIPACE_BEAMDEPOSITCURRENTINNER_H_

#include "particles/ShapeFactors.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "Hipace.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

/** \brief Loop over beam particles in iterator (=box) pti and deposit their current
 * into jx_fab, jy_fab, and jz_fab
 *
 * Only deposit charge and current for beam particles in islice.
 * Note that particles are not sorted in memory, argument bins only contains indices of particles
 * sorted per slice.
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the deposition
 * \tparam depos_order_z Order of the longitudinal shape factor for the deposition
 * \param[in] ptile particle tile, contains data of all particles in a box
 * \param[in,out] isl_fab array containing the fields, on the box corresponding to ptile
 * \param[in] jxb_cmp field component of jx for the beam
 * \param[in] jyb_cmp field component of jy for the beam
 * \param[in] jzb_cmp field component of jz for the beam
 * \param[in] rhob_cmp fddield component of rho for the beam
 * \param[in] dx cell size in each dimension
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 * \param[in] q elementary charge
 * \param[in] islice Particles in slice islice will deposit
 * \param[in] bins Indices of particles arranged per slices.
 * \param[in] box_offset offset to particles on this box.
 * \param[in] do_beam_jx_jy_deposition whether the beams deposit Jx and Jy
 * \param[in] which_slice defines if this or the next slice is handled
 * \param[in] nghost number of ghost particles, all at the end of the particle array.
 *            Use for depositing transverse currents in the Next slice when processing
 *            islice = 0.
 */
template <int depos_order_xy, int depos_order_z>
void doDepositionShapeN (const BeamParticleContainer& ptile,
                         amrex::FArrayBox& isl_fab,
                         const int jxb_cmp,
                         const int jyb_cmp,
                         const int jzb_cmp,
                         const int rhob_cmp,
                         amrex::Real const * const AMREX_RESTRICT dx,
                         amrex::Real x_pos_offset,
                         amrex::Real y_pos_offset,
                         amrex::Real z_pos_offset,
                         amrex::Real const q,
                         int islice,
                         const BeamBins& bins,
                         int box_offset,
                         const bool do_beam_jx_jy_deposition,
                         const int which_slice,
                         int nghost)
{
    using namespace amrex::literals;

    // Whether the ghost slice has to be deposited
    const bool deposit_ghost = ((which_slice==WhichSlice::Next) && (islice == 0));
    if (deposit_ghost && !do_beam_jx_jy_deposition) return;

    // Ghost particles are indexed [ptile.numParticles()-nghost, ptile.numParticles()-1]
    if (deposit_ghost) box_offset = ptile.numParticles()-nghost;

    PhysConst const phys_const = get_phys_const();

    // Extract particle properties
    const auto& aos = ptile.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin() + box_offset;
    const auto& soa = ptile.GetStructOfArrays(); // For momenta and weights
    const auto  wp = soa.GetRealData(BeamIdx::w).data() + box_offset;
    const auto uxp = soa.GetRealData(BeamIdx::ux).data() + box_offset;
    const auto uyp = soa.GetRealData(BeamIdx::uy).data() + box_offset;
    const auto uzp = soa.GetRealData(BeamIdx::uz).data() + box_offset;

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = Hipace::m_normalized_units? 1. : dxi*dyi*dzi;

    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

    Array3<amrex::Real> const isl_arr = isl_fab.array();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(isl_fab.box().ixType().cellCentered(),
        "jx, jy, jz and rho must be nodal in all directions.");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(isl_fab.box().length(2) == 1,
        "jx, jy, jz and rho must be exactly one cell thick in the z direction.");

    BeamBins::index_type const * const indices = bins.permutationPtr();
    BeamBins::index_type const * const offsets = bins.offsetsPtrCpu();
    BeamBins::index_type cell_start = 0, cell_stop = 0;

    // The particles that are in slice islice are
    // given by the indices[cell_start:cell_stop]
    if (which_slice == WhichSlice::This) {
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
        num_particles,
        [=] AMREX_GPU_DEVICE (long idx) {
            // Particles in the same slice must be accessed through the bin sorter.
            // Ghost particles are simply contiguous in memory.
            const int ip = deposit_ghost ? cell_start+idx : indices[cell_start+idx];

            // Skip invalid particles and ghost particles not in the last slice
            if (pos_structs[ip].id() < 0) return;
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
            const amrex::Real xmid = (pos_structs[ip].pos(0) - x_pos_offset)*dxi;
            // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid);

            // y direction
            const amrex::Real ymid = (pos_structs[ip].pos(1) - y_pos_offset)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid);

            // z direction
            const amrex::Real zmid = (pos_structs[ip].pos(2) - z_pos_offset)*dzi;
            amrex::Real sz_cell[depos_order_z + 1]; // depos_order_z MUST be 0.
            int l_cell = compute_shape_factor<depos_order_z>(sz_cell, zmid);
            l_cell = 0;
            amrex::ignore_unused(l_cell);

            // Deposit current into jx, jy, jz, rho
            for (int iz=0; iz<=depos_order_z; iz++){
                for (int iy=0; iy<=depos_order_xy; iy++){
                    for (int ix=0; ix<=depos_order_xy; ix++){
                        if (jxb_cmp != -1) { // do_beam_jx_jy_deposition
                            amrex::Gpu::Atomic::Add(
                                isl_arr.ptr(j_cell+ix, k_cell+iy, jxb_cmp),
                                sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqx);
                            amrex::Gpu::Atomic::Add(
                                isl_arr.ptr(j_cell+ix, k_cell+iy, jyb_cmp),
                                sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqy);
                        }
                        if (jzb_cmp != -1) { // do_beam_jz_deposition
                            amrex::Gpu::Atomic::Add(
                                isl_arr.ptr(j_cell+ix, k_cell+iy, jzb_cmp),
                                sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqz);
                        }
                        if (rhob_cmp != -1) { // do_beam_rho_deposition
                            amrex::Gpu::Atomic::Add(
                                isl_arr.ptr(j_cell+ix, k_cell+iy, rhob_cmp),
                                sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wq);
                        }
                    }
                }
            }
        }
        );
}

#endif // BEAMDEPOSITCURRENTINNER_H_
