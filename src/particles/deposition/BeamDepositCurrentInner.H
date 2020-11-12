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
#include "Constants.H"
#include "Hipace.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

/** \brief Loop over beam particles in iterator (=box) pti and deposit their current
 * into jx_fab, jy_fab and jz_fab and their charge density to rho_fab
 *
 * If slice_beam is true, only deposit charge and current for beam particles in islice.
 * Note that particles are not sorted in memory, argument bins only contains indices of particles
 * sorted per slice.
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the deposition
 * \tparam depos_order_z Order of the longitudinal shape factor for the deposition
 * \param[in] pti particle iterator, contains data of all particles in a box
 * \param[in,out] jx_fab array of current density jx, on the box corresponding to pti
 * \param[in,out] jy_fab array of current density jy, on the box corresponding to pti
 * \param[in,out] jz_fab array of current density jz, on the box corresponding to pti
 * \param[in,out] rho_fab array of charge density rho, on the box corresponding to pti
 * \param[in] dx cell size in each dimension
 * \param[in] xyzmin lower corner of the box, in physical space
 * \param[in] lo lower corner of the box, in index space
 * \param[in] q elementary charge
 * \param[in] islice If slice_beam true, particles in slice islice will deposit
 * \param[in] bins Indices of particles arranged per slices. Used if slice_beam is true.
 * \param[in] slice_beam Whether to deposit in 3D array or in slice islice only.
 */
template <int depos_order_xy, int depos_order_z>
void doDepositionShapeN (const BeamParticleIterator& pti,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         amrex::FArrayBox& rho_fab,
                         amrex::Real const * const AMREX_RESTRICT dx,
                         amrex::Real const * const AMREX_RESTRICT xyzmin,
                         amrex::Dim3 const lo,
                         amrex::Real const q,
                         int islice,
                         amrex::DenseBins<BeamParticleContainer::ParticleType>& bins,
                         bool slice_beam)
{
    using namespace amrex::literals;

    PhysConst const phys_const = get_phys_const();

    // Extract particle properties
    const auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    const auto& soa = pti.GetStructOfArrays(); // For momenta and weights
    const auto  wp = soa.GetRealData(BeamIdx::w).data();
    const auto uxp = soa.GetRealData(BeamIdx::ux).data();
    const auto uyp = soa.GetRealData(BeamIdx::uy).data();
    const auto uzp = soa.GetRealData(BeamIdx::uz).data();

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = Hipace::m_normalized_units? 1. : dxi*dyi*dzi;
    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    const amrex::Real zmin = xyzmin[2];

    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::Array4<amrex::Real> const& rho_arr = rho_fab.array();

    constexpr int CELL = amrex::IndexType::CELL;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        jx_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jy_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jz_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        rho_fab.box().type() == amrex::IntVect(CELL, CELL, CELL),
        "jx, jy, jz and rho must be nodal in all directions."
        );

    amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type*
        indices = nullptr;
    amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const * offsets = 0;
    amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type cell_start = 0, cell_stop = 0;
    if (slice_beam){
        indices = bins.permutationPtr();
        offsets = bins.offsetsPtr();
        // The particles that are in slice islice are
        // given by the indices[cell_start:cell_stop]
        cell_start = offsets[islice];
        cell_stop  = offsets[islice+1];
    }
    int const num_particles = slice_beam ? cell_stop-cell_start : pti.numParticles();

    // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
    amrex::ParallelFor(
        num_particles,
        [=] AMREX_GPU_DEVICE (long idx) {
            const int ip = slice_beam ? indices[cell_start+idx] : idx;
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
            const amrex::Real xmid = (pos_structs[ip].pos(0) - xmin)*dxi;
            // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid - 0.5_rt);

            // y direction
            const amrex::Real ymid = (pos_structs[ip].pos(1) - ymin)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid - 0.5_rt);

            // z direction
            const amrex::Real zmid = (pos_structs[ip].pos(2) - zmin)*dzi;
            amrex::Real sz_cell[depos_order_z + 1];
            int l_cell = compute_shape_factor<depos_order_z>(sz_cell, zmid - 0.5_rt);
            if (slice_beam) { l_cell = 0; }

            // Deposit current into jx_arr, jy_arr, jz_arr, and rho_arr
            for (int iz=0; iz<=depos_order_z; iz++){
                for (int iy=0; iy<=depos_order_xy; iy++){
                    for (int ix=0; ix<=depos_order_xy; ix++){
                        amrex::Gpu::Atomic::Add(
                            &jx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                            sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqx);
                        amrex::Gpu::Atomic::Add(
                            &jy_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                            sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqy);
                        amrex::Gpu::Atomic::Add(
                            &jz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                            sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqz);
                        if (!slice_beam){
                            amrex::Gpu::Atomic::Add(
                                &rho_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                                sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wq);
                        }
                    }
                }
            }
        }
        );
}

#endif // BEAMDEPOSITCURRENTINNER_H_
