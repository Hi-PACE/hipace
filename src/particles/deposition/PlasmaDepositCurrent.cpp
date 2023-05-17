/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaDepositCurrent.H"

#include "particles/particles_utils/ShapeFactors.H"
#include "particles/particles_utils/FieldGather.H"
#include "particles/plasma/PlasmaParticleContainer.H"
#include "particles/sorting/TileSort.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"


void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields, const MultiLaser& multi_laser,
                const int which_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                const bool deposit_chi, const bool deposit_rhomjz,
                amrex::Vector<amrex::Geometry> const& gm, int const lev,
                const PlasmaBins& bins, int bin_size)
{
    HIPACE_PROFILE("DepositCurrent_PlasmaParticleContainer()");
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next ||
    which_slice == WhichSlice::RhomJzIons || which_slice == WhichSlice::Salame,
    "Current deposition can only be done in this slice (WhichSlice::This), the next slice "
    " (WhichSlice::Next), for the ion charge deposition (WhichSLice::RhomJzIons)"
    " or for the Salame slice (WhichSlice::Salame)");

    const amrex::Real max_qsa_weighting_factor = plasma.m_max_qsa_weighting_factor;
    const amrex::Real charge = (which_slice == WhichSlice::RhomJzIons) ? -plasma.m_charge : plasma.m_charge;
    const amrex::Real mass = plasma.m_mass;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {
        // Extract the fields currents
        // Do not access the field if the kernel later does not deposit into it,
        // the field might not be allocated. Use 0 as dummy component instead
        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];
        const int     jx_cmp = deposit_jx_jy  ? Comps[which_slice]["jx"]     : -1;
        const int     jy_cmp = deposit_jx_jy  ? Comps[which_slice]["jy"]     : -1;
        const int     jz_cmp = deposit_jz     ? Comps[which_slice]["jz"]     : -1;
        const int    rho_cmp = deposit_rho    ? Comps[which_slice]["rho"]    : -1;
        const int    chi_cmp = deposit_chi    ? Comps[which_slice]["chi"]    : -1;
        const int rhomjz_cmp = deposit_rhomjz ? Comps[which_slice]["rhomjz"] : -1;

        amrex::Vector<amrex::FArrayBox>& tmp_dens = fields.getTmpDensities();

        // extract the laser Fields
        const amrex::MultiFab& a_mf = multi_laser.getSlices();

        // Offset for converting positions to indexes
        const amrex::Real x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

        // Extract particle properties
        auto& aos = pti.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin();
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        amrex::Real * const AMREX_RESTRICT wp = soa.GetRealData(PlasmaIdx::w).data();
        const amrex::Real * const AMREX_RESTRICT uxp = soa.GetRealData(PlasmaIdx::ux).data();
        const amrex::Real * const AMREX_RESTRICT uyp = soa.GetRealData(PlasmaIdx::uy).data();
        const amrex::Real * const AMREX_RESTRICT psip = soa.GetRealData(PlasmaIdx::psi).data();

        int const * const AMREX_RESTRICT a_ion_lev =
            plasma.m_can_ionize ? soa.GetIntData(PlasmaIdx::ion_lev).data() : nullptr;

        // Extract laser array from MultiFab
        const Array3<const amrex::Real> a_laser_arr =
            multi_laser.m_use_laser ? a_mf[pti].const_array(WhichLaserSlice::n00j00_r)
                                    : amrex::Array4<const amrex::Real>();

        // Extract box properties
        const amrex::Real dx_inv = gm[lev].InvCellSize(0);
        const amrex::Real dy_inv = gm[lev].InvCellSize(1);
        const amrex::Real dz_inv = gm[lev].InvCellSize(2);
        // in normalized units this is rescaling dx and dy for MR,
        // while in SI units it's the factor for charge to charge density
        const amrex::Real invvol = Hipace::m_normalized_units ?
            gm[0].CellSize(0)*gm[0].CellSize(1)*dx_inv*dy_inv
            : dx_inv*dy_inv*dz_inv;

        const PhysConst pc = get_phys_const();
        const amrex::Real clight = pc.c;
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

        const bool do_tiling = Hipace::m_do_tiling;

        Array3<amrex::Real> const isl_arr =
            do_tiling ? tmp_dens[ithread].array() : isl_fab.array();
        const int jx     = (do_tiling && jx_cmp     != -1) ? 0 : jx_cmp;
        const int jy     = (do_tiling && jy_cmp     != -1) ? 1 : jy_cmp;
        const int jz     = (do_tiling && jz_cmp     != -1) ? 2 : jz_cmp;
        const int rho    = (do_tiling && rho_cmp    != -1) ? 3 : rho_cmp;
        const int chi    = (do_tiling && chi_cmp    != -1) ? 4 : chi_cmp;
        const int rhomjz = (do_tiling && rhomjz_cmp != -1) ? 5 : rhomjz_cmp;

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
        for (int a_itile=0; a_itile<ntiles; a_itile++){

#ifndef AMREX_USE_GPU
            if (do_tiling) tmp_dens[ithread].setVal(0.);
#endif
            // Get the x and y indices of current tile from its linearized index itile = itiley + itilex * ntiley
            const int itilex = do_tiling ? a_itile / ntiley : 0;
            const int itiley = do_tiling ? a_itile % ntiley : 0;
            PlasmaBins::index_type const * const a_indices = do_tiling ? bins.permutationPtr() : nullptr;
            PlasmaBins::index_type const * const a_offsets = do_tiling ? bins.offsetsPtr() : nullptr;
            const int a_itilex_bs = do_tiling ? itilex * bin_size : 0;
            const int a_itiley_bs = do_tiling ? itiley * bin_size : 0;

            int num_particles = do_tiling ? a_offsets[a_itile+1]-a_offsets[a_itile]
                                          : pti.numParticles();

            if (Hipace::m_outer_depos_loop) {
                num_particles *= (Hipace::m_depos_order_xy + 1);
            }

            // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
            amrex::ParallelFor(
                amrex::TypeList<
                    amrex::CompileTimeOptions<0, 1, 2, 3>,  // depos_order
                    amrex::CompileTimeOptions<false, true>, // outer_depos_loop
                    amrex::CompileTimeOptions<false, true>, // can_ionize
#ifdef AMREX_USE_GPU
                    amrex::CompileTimeOptions<false>,       // do_tiling (disabled on GPU)
#else
                    amrex::CompileTimeOptions<false, true>, // do_tiling
#endif
                    amrex::CompileTimeOptions<false, true>  // use_laser
                >{}, {
                    Hipace::m_depos_order_xy,
                    Hipace::m_outer_depos_loop,
                    plasma.m_can_ionize,
                    do_tiling,
                    multi_laser.m_use_laser
                },
                num_particles,
                [=] AMREX_GPU_DEVICE (int idx, auto depos_order, auto outer_depos_loop,
                    auto can_ionize, auto c_do_tiling, auto use_laser) noexcept {
                constexpr int depos_order_xy = depos_order.value;
                // Using 1 thread per particle and per deposited cell is only done in the fast (x) direction.
                // This can also be applied in the y direction, but so far does not show significant gain.
                constexpr bool outer_depos_loop_x = outer_depos_loop.value;
                constexpr int outer_depos_order_x_1 = outer_depos_loop_x ? (depos_order_xy + 1) : 1;
                constexpr int inner_depos_order_x = outer_depos_loop_x ? 0 : depos_order_xy;

                int ip = idx / outer_depos_order_x_1;

                [[maybe_unused]] auto indices = a_indices;
                [[maybe_unused]] auto offsets = a_offsets;
                [[maybe_unused]] auto itile = a_itile;
                if constexpr (c_do_tiling.value) {
                    ip = indices[offsets[itile]+ip];
                }

                const int ox = idx % outer_depos_order_x_1;

                const auto positions = pos_structs[ip];
                // only deposit plasma currents on or below their according MR level
                if (positions.id() < 0 || positions.cpu() < lev) return;
                const amrex::Real psi_inv = 1._rt/psip[ip];
                const amrex::Real xp = positions.pos(0);
                const amrex::Real yp = positions.pos(1);
                const amrex::Real vx_c = uxp[ip] * psi_inv;
                const amrex::Real vy_c = uyp[ip] * psi_inv;

                // calculate charge of the plasma particles
                amrex::Real q_invvol = charge_invvol;
                amrex::Real q_mu0_mass_ratio = charge_mu0_mass_ratio;
                [[maybe_unused]] auto ion_lev = a_ion_lev;
                if constexpr (can_ionize.value) {
                    q_invvol *= ion_lev[ip];
                    q_mu0_mass_ratio *= ion_lev[ip];
                }

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                amrex::Real Aabssqp = 0._rt;
                [[maybe_unused]] auto laser_arr = a_laser_arr;
                if constexpr (use_laser.value) {
                    doLaserGatherShapeN<depos_order_xy>(xp, yp, Aabssqp, laser_arr,
                                                        dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                }

                // calculate gamma/psi for plasma particles
                const amrex::Real gamma_psi = 0.5_rt * (
                    (1._rt + 0.5_rt * Aabssqp) * psi_inv * psi_inv // TODO: fix units
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
                        auto [shape_x, cell_x] =
                            compute_single_shape_factor<outer_depos_loop_x, depos_order_xy>(xmid, tx);

                        // y direction
                        auto [shape_y, cell_y] =
                            compute_single_shape_factor<false, depos_order_xy>(ymid, iy);

                        [[maybe_unused]] auto itilex_bs = a_itilex_bs;
                        [[maybe_unused]] auto itiley_bs = a_itiley_bs;
                        if constexpr (c_do_tiling.value) {
                            cell_x -= itilex_bs;
                            cell_y -= itiley_bs;
                        }

                        const amrex::Real charge_density = q_invvol * wp[ip] * shape_x * shape_y;
                        // wqx, wqy wqz are particle current in each direction
                        const amrex::Real wqx     = charge_density * vx_c;
                        const amrex::Real wqy     = charge_density * vy_c;
                        const amrex::Real wqz     = charge_density * (gamma_psi-1._rt) * clight;
                        const amrex::Real wq      = charge_density * gamma_psi;
                        const amrex::Real wchi    = charge_density * q_mu0_mass_ratio * psi_inv;
                        const amrex::Real wrhomjz = charge_density;

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
                        if (rhomjz != -1) { // deposit_rhomjz
                            amrex::Gpu::Atomic::Add(isl_arr.ptr(cell_x, cell_y, rhomjz), wrhomjz);
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
                    isl_fab.atomicAdd(tmp_dens[ithread], srcbx, dstbx, 0, jx_cmp, 1);
                    isl_fab.atomicAdd(tmp_dens[ithread], srcbx, dstbx, 1, jy_cmp, 1);
                }
                if (jz_cmp != -1) {
                    isl_fab.atomicAdd(tmp_dens[ithread], srcbx, dstbx, 2, jz_cmp, 1);
                }
                if (rho_cmp != -1) {
                    isl_fab.atomicAdd(tmp_dens[ithread], srcbx, dstbx, 3, rho_cmp, 1);
                }
                if (chi_cmp != -1) {
                    isl_fab.atomicAdd(tmp_dens[ithread], srcbx, dstbx, 4, chi_cmp, 1);
                }
                if (rhomjz_cmp != -1) {
                    isl_fab.atomicAdd(tmp_dens[ithread], srcbx, dstbx, 5, rhomjz_cmp, 1);
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
}
