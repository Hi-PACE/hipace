/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_PLASMADEPOSITCURRENTINNER_H_
#define HIPACE_PLASMADEPOSITCURRENTINNER_H_

#include "particles/particles_utils/ShapeFactors.H"
#include "particles/sorting/TileSort.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
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
 * \tparam outer_depos_loop if the loop over depos_order is included in the loop over particles
 * \tparam depos_order_xy Order of the transverse shape factor for the deposition
 * \tparam t_use_laser whether a laser can be used
 * \tparam t_do_tiling whether tiling can be used
 * \tparam t_can_ionize whether the plasma could be ionized
 * \param[in] use_laser whether a laser is used or not
 * \param[in] do_tiling whether to use tiling
 * \param[in] can_ionize whether charge needs to be multiplied by ion_lev
 * \param[in] pti particle iterator, contains data of all particles in a box
 * \param[in,out] isl_fab array of current slice, on the box corresponding to pti
 * \param[in] jx_cmp array component of current density jx
 * \param[in] jy_cmp array component of current density jy
 * \param[in] jz_cmp array component of current density jz
 * \param[in] rho_cmp array component of charge density rho
 * \param[in] chi_cmp array component of current density chi
 * \param[in] a_mf MultiFab containing the normalized vector potential of the laser a
 * \param[in] tmp_densities tmp FABs where densities are deposited to avoid race conditions with tiling
 * \param[in] dx cell size in each dimension
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 * \param[in] charge of a particle
 * \param[in] mass of a particle
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
 * \param[in] max_qsa_weighting_factor maximum allowed weighting factor gamma/(Psi+1)
 * \param[in] bins objects containing indices of plasma particles in each tile
 * \param[in] bin_size tile size (square)
 */
template <bool outer_depos_loop, int depos_order_xy, bool t_use_laser, bool t_do_tiling, bool t_can_ionize>
void doDepositionShapeN (const bool use_laser,
                         const bool do_tiling,
                         const bool can_ionize,
                         const PlasmaParticleIterator& pti,
                         amrex::FArrayBox& isl_fab,
                         const int jx_cmp,
                         const int jy_cmp,
                         const int jz_cmp,
                         const int rho_cmp,
                         const int chi_cmp,
                         const amrex::MultiFab& a_mf,
                         amrex::Vector<amrex::FArrayBox>& tmp_densities,
                         amrex::Real const * const AMREX_RESTRICT dx,
                         amrex::Real const x_pos_offset,
                         amrex::Real const y_pos_offset,
                         amrex::Real const z_pos_offset,
                         amrex::Real const charge,
                         amrex::Real const mass,
                         const bool temp_slice,
                         const amrex::Real max_qsa_weighting_factor,
                         const PlasmaBins& bins, int bin_size)
{
    using namespace amrex::literals;

    // Using 1 thread per particle and per deposited cell is only done in the fast (x) direction.
    // This can also be applied in the y direction, but so far does not show significant gain.
    static constexpr bool outer_depos_loop_x = outer_depos_loop;
    static constexpr int outer_depos_order_x_1 = outer_depos_loop_x ? (depos_order_xy + 1) : 1;
    static constexpr int inner_depos_order_x = outer_depos_loop_x ? 0 : depos_order_xy;

    // Extract particle properties
    auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    auto& soa = pti.GetStructOfArrays(); // For momenta and weights

    amrex::Real * const AMREX_RESTRICT wp = soa.GetRealData(PlasmaIdx::w).data();
    const amrex::Real * const AMREX_RESTRICT uxp =
        soa.GetRealData(temp_slice ? PlasmaIdx::ux_temp : PlasmaIdx::ux).data();
    const amrex::Real * const AMREX_RESTRICT uyp =
        soa.GetRealData(temp_slice ? PlasmaIdx::uy_temp : PlasmaIdx::uy).data();
    const amrex::Real * const AMREX_RESTRICT psip =
        soa.GetRealData(temp_slice ? PlasmaIdx::psi_temp : PlasmaIdx::psi).data();

    int const * const AMREX_RESTRICT ion_lev =
        can_ionize ? soa.GetIntData(PlasmaIdx::ion_lev).data() : nullptr;

    // Extract laser array from MultiFab
    const amrex::Array4<const amrex::Real> a_arr =
        use_laser ? a_mf[pti].array() : amrex::Array4<const amrex::Real>();
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    // Extract box properties
    const amrex::Real invvol = Hipace::m_normalized_units ? 1._rt : 1._rt/(dx[0]*dx[1]*dx[2]);
    const amrex::Real dx_inv = 1._rt/dx[0];
    const amrex::Real dy_inv = 1._rt/dx[1];

    const PhysConst pc = get_phys_const();
    const amrex::Real clightinv = 1.0_rt/pc.c;
    const amrex::Real charge_invvol = charge * invvol;
    const amrex::Real charge_mu0_mass_ratio = charge * pc.mu0 / mass;

    int n_qsa_violation = 0;
    amrex::Gpu::DeviceScalar<int> gpu_n_qsa_violation(n_qsa_violation);
    int* const AMREX_RESTRICT p_n_qsa_violation = gpu_n_qsa_violation.dataPtr();


#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    {
        const int ithread = omp_get_thread_num();
#else
        const int ithread = 0;
#endif
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(isl_fab.box().ixType().cellCentered(),
            "jx, jy, jz, and rho must be nodal in all directions.");

        Array3<amrex::Real> const isl_arr =
            do_tiling ? tmp_densities[ithread].array() : isl_fab.array();
        const int jx  = (do_tiling && jx_cmp  != -1) ? 0 : jx_cmp;
        const int jy  = (do_tiling && jy_cmp  != -1) ? 1 : jy_cmp;
        const int jz  = (do_tiling && jz_cmp  != -1) ? 2 : jz_cmp;
        const int rho = (do_tiling && rho_cmp != -1) ? 3 : rho_cmp;
        const int chi = (do_tiling && chi_cmp != -1) ? 4 : chi_cmp;

        int ntiley = 0;
        if (do_tiling) {
            const int ng = Fields::m_slices_nguards[0];
            const int ncellx = isl_fab.box().bigEnd(0)-isl_fab.box().smallEnd(0)+1-2*ng;
            const int ncelly = isl_fab.box().bigEnd(1)-isl_fab.box().smallEnd(1)+1-2*ng;
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
            // Get the x and y indices of current tile from its linearized index itile = itiley + itilex * ntiley
            const int itilex = do_tiling ? itile / ntiley : 0;
            const int itiley = do_tiling ? itile % ntiley : 0;
            PlasmaBins::index_type const * const indices = do_tiling ? bins.permutationPtr() : nullptr;
            PlasmaBins::index_type const * const offsets = do_tiling ? bins.offsetsPtr() : nullptr;
            const int itilex_bs = do_tiling ? itilex * bin_size : 0;
            const int itiley_bs = do_tiling ? itiley * bin_size : 0;

            const int num_particles = do_tiling ? offsets[itile+1]-offsets[itile] : pti.numParticles();

            // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
            amrex::ParallelFor(outer_depos_order_x_1 * num_particles,
                [   // explicit lambda capture is required here
                    pos_structs,
                    wp, uxp, uyp, psip,
                    max_qsa_weighting_factor, p_n_qsa_violation,
                    clightinv, x_pos_offset, y_pos_offset, invvol, dx_inv, dy_inv,
                    charge_invvol, charge_mu0_mass_ratio,
                    isl_arr,
                    jx, jy, jz, rho, chi,
                    // tiling
                    indices, offsets, itile, itilex_bs, itiley_bs,
                    // laser
                    a_arr, dx_arr, z_pos_offset,
                    // ionization
                    ion_lev
                ] AMREX_GPU_DEVICE (int idx) noexcept {
                int ip = idx / outer_depos_order_x_1;

                if constexpr (t_do_tiling) {
                    if (indices) { // do_tiling
                        ip = indices[offsets[itile]+ip];
                    }
                }

                const int ox = idx % outer_depos_order_x_1;

                const auto positions = pos_structs[ip];
                if (positions.id() < 0) return;
                const amrex::Real psi = psip[ip];
                const amrex::Real xp = positions.pos(0);
                const amrex::Real yp = positions.pos(1);
                const amrex::Real vx_c = uxp[ip] / psi;
                const amrex::Real vy_c = uyp[ip] / psi;

                // calculate charge of the plasma particles
                amrex::Real q_invvol = charge_invvol;
                amrex::Real q_mu0_mass_ratio = charge_mu0_mass_ratio;
                if constexpr (t_can_ionize) {
                    if (ion_lev) { // can_ionize
                        q_invvol *= ion_lev[ip];
                        q_mu0_mass_ratio *= ion_lev[ip];
                    }
                }

                amrex::Real Aabssqp = 0._rt;
                if constexpr (t_use_laser) {
                    if (a_arr) { // use_laser
                        doLaserGatherShapeN<depos_order_xy,0> (xp, yp, 0._rt, // zp not used
                                            Aabssqp, a_arr, dx_arr,
                                            x_pos_offset, y_pos_offset, z_pos_offset);
                    }
                }

                // calculate gamma/psi for plasma particles
                const amrex::Real gamma_psi = 0.5_rt * (
                    (1._rt + 0.5_rt * Aabssqp) / (psi * psi) // TODO: fix units
                    + vx_c * vx_c * clightinv * clightinv
                    + vy_c * vy_c * clightinv * clightinv
                    + 1._rt
                );

                if ((gamma_psi < 0.0_rt || gamma_psi > max_qsa_weighting_factor) && ox == 0)
                {
                    // This particle violates the QSA, discard it and do not deposit its current
                    amrex::Gpu::Atomic::Add(p_n_qsa_violation, 1);
                    wp[ip] = 0.0_rt;
                    pos_structs[ip].id() = -std::abs(positions.id());
                    return;
                }

                for (int iy=0; iy <= depos_order_xy; ++iy) {
                    for (int ix=0; ix <= inner_depos_order_x; ++ix) {
                        int tx = 0;
                        if constexpr (outer_depos_loop_x) {
                            tx = ox;
                        } else {
                            tx = ix;
                        }
                        // --- Compute shape factors
                        // x direction
                        const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                        auto [shape_x, cell_x] =
                            compute_single_shape_factor<outer_depos_loop_x, depos_order_xy>(xmid, tx);

                        // y direction
                        const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;
                        auto [shape_y, cell_y] =
                            compute_single_shape_factor<false, depos_order_xy>(ymid, iy);

                        if constexpr (t_do_tiling) {
                            if (indices) { // do_tiling
                                cell_x -= itilex_bs;
                                cell_y -= itiley_bs;
                            }
                        }

                        const amrex::Real charge_density = q_invvol * wp[ip] * shape_x * shape_y;
                        // wqx, wqy wqz are particle current in each direction
                        const amrex::Real wqx  = charge_density * vx_c;
                        const amrex::Real wqy  = charge_density * vy_c;
                        const amrex::Real wqz  = charge_density * (gamma_psi-1._rt) / clightinv;
                        const amrex::Real wq   = charge_density * gamma_psi;
                        const amrex::Real wchi = charge_density * q_mu0_mass_ratio / psi;

                        // Deposit current into isl_arr
                        if (jx != -1) { // deposit_jx_jy
                            amrex::Gpu::Atomic::Add(isl_arr.ptr(cell_x, cell_y, jx), wqx);
                            amrex::Gpu::Atomic::Add(isl_arr.ptr(cell_x, cell_y, jy), wqy);
                        }
                        if (jz != -1) { // deposit_jz
                            amrex::Gpu::Atomic::Add(isl_arr.ptr(cell_x, cell_y, jz), wqz);
                        }
                        if (rho != -1) { // deposit_rho
                            amrex::Gpu::Atomic::Add(isl_arr.ptr(cell_x, cell_y, rho), wq);
                        }
                        if (chi != -1) { // deposit_chi
                            amrex::Gpu::Atomic::Add(isl_arr.ptr(cell_x, cell_y, chi), wchi);
                        }
                    }
                }
            });
#ifndef AMREX_USE_GPU
            if (do_tiling) {
                // If tiling is on, the current was deposited (see above) in temporary tile arrays.
                // Now, we atomic add from these temporary arrays to the main arrays
                amrex::Box srcbx = {{0, 0, 0}, {bin_size-1, bin_size-1, 0}};
                amrex::Box dstbx = {{itilex*bin_size, itiley*bin_size, pti.tilebox().smallEnd(2)},
                                    {(itilex+1)*bin_size-1, (itiley+1)*bin_size-1, pti.tilebox().smallEnd(2)}};
                srcbx.grow(Fields::m_slices_nguards);
                dstbx.grow(Fields::m_slices_nguards);
                if (jx_cmp != -1) {
                    isl_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 0, jx_cmp, 1);
                    isl_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 1, jy_cmp, 1);
                }
                if (jz_cmp != -1) {
                    isl_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 2, jz_cmp, 1);
                }
                if (rho_cmp != -1) {
                    isl_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 3, rho_cmp, 1);
                }
                if (chi_cmp != -1) {
                    isl_fab.atomicAdd(tmp_densities[ithread], srcbx, dstbx, 4, chi_cmp, 1);
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
