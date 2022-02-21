/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_PLASMADEPOSITCURRENTINNER_H_
#define HIPACE_PLASMADEPOSITCURRENTINNER_H_

#include "particles/ShapeFactors.H"
#include "particles/TileSort.H"
#include "utils/Constants.H"
#include "Hipace.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

/** \brief Loop over plasma particles in iterator (=box) pti and deposit their current
 * into jx_fab, jy_fab and jz_fab and their density to rho_fab
 *
 * current_depo_type affects the general behavior of this function:
 * - If current_depo_type == WhichSlice::This, deposit jx, jy, jz and rho
 *   with current values of transverse position, wp, uxp, uyp and psip.
 * - If current_depo_type == WhichSlice::Next, deposit jx  and jy only,
 *   with projected values of transverse position, wp, uxp, uyp and psip stored in temp arrays.
 *
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the deposition
 * \tparam depos_order_z Order of the longitudinal shape factor for the deposition
 * \param[in] pti particle iterator, contains data of all particles in a box
 * \param[in,out] jx_fab array of current density jx, on the box corresponding to pti
 * \param[in,out] jy_fab array of current density jy, on the box corresponding to pti
 * \param[in,out] jz_fab array of current density jz, on the box corresponding to pti
 * \param[in,out] rho_fab array of charge density rho, on the box corresponding to pti
 * \param[in,out] jxx_fab array of current density jxx, on the box corresponding to pti
 * \param[in,out] jxy_fab array of current density jxy, on the box corresponding to pti
 * \param[in,out] jyy_fab array of current density jyy, on the box corresponding to pti
 * \param[in] tmp_densities tmp FABs where densities are deposited to avoid race conditions with tiling
 * \param[in] dx cell size in each dimension
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] charge of a particle
 * \param[in] can_ionize whether charge needs to be multiplied by ion_lev
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
 * \param[in] deposit_jx_jy if true, deposit to jx and jy
 * \param[in] deposit_jz if true, deposit to jz
 * \param[in] deposit_j_squared if true, deposit jxx, jxy and jyy
 * \param[in] deposit_rho if true, deposit to rho
 * \param[in] max_qsa_weighting_factor maximum allowed weighting factor gamma/(Psi+1)
 * \param[in] bins objects containing indices of plasma particles in each tile
 * \param[in] bin_size tile size (square)
 */
template <int depos_order_xy, int depos_order_z>
void doDepositionShapeN (const PlasmaParticleIterator& pti,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         amrex::FArrayBox& rho_fab,
                         amrex::FArrayBox& jxx_fab,
                         amrex::FArrayBox& jxy_fab,
                         amrex::FArrayBox& jyy_fab,
                         amrex::Vector<amrex::FArrayBox>& tmp_densities,
                         amrex::Real const * const AMREX_RESTRICT dx,
                         amrex::Real const x_pos_offset,
                         amrex::Real const y_pos_offset,
                         amrex::Real const charge,
                         const bool can_ionize,
                         const bool temp_slice,
                         const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                         const bool deposit_j_squared, const amrex::Real max_qsa_weighting_factor,
                         PlasmaBins& bins, int bin_size)
{
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    const bool do_tiling = Hipace::m_do_tiling;

    // Extract particle properties
    auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    auto& soa = pti.GetStructOfArrays(); // For momenta and weights

    amrex::Real * const wp = soa.GetRealData(PlasmaIdx::w).data();
    int * const ion_lev = soa.GetIntData(PlasmaIdx::ion_lev).data();
    const amrex::Real * const uxp = (!temp_slice) ?
        soa.GetRealData(PlasmaIdx::ux).data() : soa.GetRealData(PlasmaIdx::ux_temp).data();
    const amrex::Real * const uyp = (!temp_slice) ?
        soa.GetRealData(PlasmaIdx::uy).data() : soa.GetRealData(PlasmaIdx::uy_temp).data();
    const amrex::Real * const psip = (!temp_slice) ?
        soa.GetRealData(PlasmaIdx::psi).data() : soa.GetRealData(PlasmaIdx::psi_temp).data();
    const amrex::Real * const const_of_motion = soa.GetRealData(PlasmaIdx::const_of_motion).data();

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = Hipace::m_normalized_units ? 1. : dxi*dyi*dzi;

    // slice is only one cell thick
    AMREX_ASSERT(pti.tilebox().smallEnd(2) == pti.tilebox().bigEnd(2));
    const int z_index = do_tiling ? 0 : pti.tilebox().smallEnd(2);

    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
    constexpr int CELL = amrex::IndexType::CELL;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        jx_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jy_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jz_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        rho_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jxx_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jxy_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jyy_fab.box().type() == amrex::IntVect(CELL, CELL, CELL),
        "jx, jy, jz, and rho must be nodal in all directions."
        );

    int n_qsa_violation = 0;
    amrex::Gpu::DeviceScalar<int> gpu_n_qsa_violation(n_qsa_violation);
    int* p_n_qsa_violation = gpu_n_qsa_violation.dataPtr();
    PlasmaBins::index_type const * const indices = do_tiling ? bins.permutationPtr() : nullptr;
    PlasmaBins::index_type const * const offsets = do_tiling ? bins.offsetsPtr() : nullptr;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    {
        const int ithread = omp_get_thread_num();
#else
        const int ithread = 0;
#endif
        amrex::Array4<amrex::Real> const& jx_arr  =
            do_tiling ? tmp_densities[ithread].array(0) : jx_fab.array();
        amrex::Array4<amrex::Real> const& jy_arr  =
            do_tiling ? tmp_densities[ithread].array(1) : jy_fab.array();
        amrex::Array4<amrex::Real> const& jz_arr  =
            do_tiling ? tmp_densities[ithread].array(2) : jz_fab.array();
        amrex::Array4<amrex::Real> const& rho_arr =
            do_tiling ? tmp_densities[ithread].array(3) : rho_fab.array();
        amrex::Array4<amrex::Real> const& jxx_arr =
            do_tiling ? tmp_densities[ithread].array(4) : jxx_fab.array();
        amrex::Array4<amrex::Real> const& jxy_arr =
            do_tiling ? tmp_densities[ithread].array(5) : jxy_fab.array();
        amrex::Array4<amrex::Real> const& jyy_arr =
            do_tiling ? tmp_densities[ithread].array(6) : jyy_fab.array();

        const int ng = Fields::m_slices_nguards[0];
        int ntiley = 0;
        if (do_tiling) {
            const int ncellx = jx_fab.box().bigEnd(0)-jx_fab.box().smallEnd(0)+1-2*ng;
            const int ncelly = jx_fab.box().bigEnd(1)-jx_fab.box().smallEnd(1)+1-2*ng;
            AMREX_ALWAYS_ASSERT(ncellx % bin_size == 0);
            AMREX_ALWAYS_ASSERT(ncelly % bin_size == 0);
            ntiley = ncelly / bin_size;
        }

        const int ntiles = do_tiling ? bins.numBins() : 1;
#ifdef AMREX_USE_OMP
#pragma omp for
#endif
        for (int itile=0; itile<ntiles; itile++){

#ifndef AMREX_USE_GPU
            if (do_tiling) tmp_densities[ithread].setVal(0.);
#endif
            const int num_particles = do_tiling ? offsets[itile+1]-offsets[itile] : pti.numParticles();
            // Get the x and y indices of current tile from its linearized index itile = itiley + itilex * ntiley
            const int itilex = do_tiling ? itile / ntiley : 0;
            const int itiley = do_tiling ? itile % ntiley : 0;

            // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
            amrex::ParallelFor(
                num_particles,
                [=] AMREX_GPU_DEVICE (long idx) {

                    const int ip = do_tiling ? indices[offsets[itile]+idx] : idx;
                    if (pos_structs[ip].id() < 0) return;

                    const amrex::Real psi = psip[ip] *
                        phys_const.q_e / (phys_const.m_e * phys_const.c * phys_const.c);
                    // calculate 1/gamma for plasma particles
                    const amrex::Real gaminv = (2.0_rt * (psi+const_of_motion[ip]) )
                                                /(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                         + uyp[ip]*uyp[ip]*clightsq
                                                         + (psi+const_of_motion[ip])*
                                                           (psi+const_of_motion[ip]));

                    if (( 1.0_rt/(gaminv*(psi+const_of_motion[ip])) < 0.0_rt) ||
                        ( 1.0_rt/(gaminv*(psi+const_of_motion[ip])) > max_qsa_weighting_factor))
                    {
                        // This particle violates the QSA, discard it and do not deposit its current
                        amrex::Gpu::Atomic::Add(p_n_qsa_violation, 1);
                        wp[ip] = 0.0_rt;
                        pos_structs[ip].id() = -std::abs(pos_structs[ip].id());
                        return;
                    }
                    // calculate plasma particle velocities
                    const amrex::Real vx = uxp[ip]*gaminv;
                    const amrex::Real vy = uyp[ip]*gaminv;
                    const amrex::Real vz = phys_const.c*(1.0_rt -(psi +const_of_motion[ip])*gaminv);

                    // calculate charge of the plasma particles
                    const amrex::Real q = can_ionize ? ion_lev[ip] * charge : charge;
                    const amrex::Real wq = q * wp[ip]/(gaminv * (psi + const_of_motion[ip]))*invvol;

                    // wqx, wqy wqz are particle current in each direction
                    const amrex::Real wqx = wq*vx;
                    const amrex::Real wqy = wq*vy;
                    const amrex::Real wqz = wq*vz;
                    const amrex::Real wqxx = q * wp[ip] * uxp[ip] * uxp[ip] * invvol
                        / ((psi+const_of_motion[ip])*(psi+const_of_motion[ip]));
                    const amrex::Real wqxy = q * wp[ip] * uxp[ip] * uyp[ip] * invvol
                        / ((psi+const_of_motion[ip])*(psi+const_of_motion[ip]));
                    const amrex::Real wqyy = q * wp[ip] * uyp[ip] * uyp[ip] * invvol
                        / ((psi+const_of_motion[ip])*(psi+const_of_motion[ip]));

                    // --- Compute shape factors
                    // x direction
                    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
                    const amrex::Real xmid = (pos_structs[ip].pos(0) - x_pos_offset)*dxi;
                    amrex::Real sx_cell[depos_order_xy + 1];
                    const int j_cell = compute_shape_factor<depos_order_xy>(
                        sx_cell, xmid - itilex*bin_size);

                    // y direction
                    const amrex::Real ymid = (pos_structs[ip].pos(1) - y_pos_offset)*dyi;
                    amrex::Real sy_cell[depos_order_xy + 1];
                    const int k_cell = compute_shape_factor<depos_order_xy>(
                        sy_cell, ymid - itiley*bin_size);
                    // Deposit current into jx_arr, jy_arr and jz_arr
                    for (int iy=0; iy<=depos_order_xy; iy++){
                        for (int ix=0; ix<=depos_order_xy; ix++){
                            if (deposit_jx_jy) {
                                amrex::Gpu::Atomic::Add(
                                    &jx_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wqx);
                                amrex::Gpu::Atomic::Add(
                                    &jy_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wqy);
                            }
                            if (deposit_jz) {
                                amrex::Gpu::Atomic::Add(
                                    &jz_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wqz);
                            }
                            if (deposit_rho) {
                                amrex::Gpu::Atomic::Add(
                                    &rho_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wq);
                            }
                            if (deposit_j_squared) {
                                amrex::Gpu::Atomic::Add(
                                    &jxx_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wqxx);
                                amrex::Gpu::Atomic::Add(
                                    &jxy_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wqxy);
                                amrex::Gpu::Atomic::Add(
                                    &jyy_arr(j_cell+ix, k_cell+iy, z_index),
                                    sx_cell[ix]*sy_cell[iy]*wqyy);
                            }
                        }
                    }
                    return;
                }
                );
#ifndef AMREX_USE_GPU
            if (do_tiling) {
                // If tiling is on, the current was deposited (see above) in temporary tile arrays.
                // Now, we atomic add from these temporary arrays to the main arrays
                amrex::Box srcbx = {{0, 0, 0}, {bin_size-1, bin_size-1, 0}};
                amrex::Box dstbx = {{itilex*bin_size, itiley*bin_size, pti.tilebox().smallEnd(2)},
                                    {(itilex+1)*bin_size-1, (itiley+1)*bin_size-1, pti.tilebox().smallEnd(2)}};
                srcbx.grow({ng, ng, 0});
                dstbx.grow({ng, ng, 0});
                if (deposit_jx_jy) {
                    jx_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 0, 0, 1);
                    jy_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 1, 0, 1);
                }
                if (deposit_jz) {
                    jz_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 2, 0, 1);
                }
                if (deposit_rho) {
                    rho_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 3, 0, 1);
                }
                if (deposit_j_squared) {
                    jxx_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 4, 0, 1);
                    jxy_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 5, 0, 1);
                    jyy_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 6, 0, 1);
                }
            }
#endif
        }
#ifdef AMREX_USE_OMP
    }
#endif

    n_qsa_violation = gpu_n_qsa_violation.dataValue();
    if (n_qsa_violation > 0 && (Hipace::m_verbose >= 3))
        amrex::Print()<< "number of QSA violating particles on this slice: " \
                      << n_qsa_violation << "\n";
}

#endif // PLASMADEPOSITCURRENTINNER_H_
