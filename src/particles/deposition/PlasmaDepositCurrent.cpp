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
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
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
    // only deposit rho individual on WhichSlice::This
    const bool deposit_rho_individual = Hipace::m_deposit_rho_individual && which_slice == WhichSlice::This;
    const std::string rho_str = deposit_rho_individual ? "rho_" + plasma.GetName() : "rho";

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {
        // Extract the fields currents
        // Do not access the field if the kernel later does not deposit into it,
        // the field might not be allocated. Use 0 as dummy component instead
        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];
        const int     jx = deposit_jx_jy  ? Comps[which_slice]["jx"]     : -1;
        const int     jy = deposit_jx_jy  ? Comps[which_slice]["jy"]     : -1;
        const int     jz = deposit_jz     ? Comps[which_slice]["jz"]     : -1;
        const int    rho = deposit_rho    ? Comps[which_slice][rho_str]  : -1;
        const int    chi = deposit_chi    ? Comps[which_slice]["chi"]    : -1;
        const int rhomjz = deposit_rhomjz ? Comps[which_slice]["rhomjz"] : -1;
        const int   aabs = Hipace::m_use_laser ? Comps[WhichSlice::This]["aabs"] : -1;

        // Offset for converting positions to indexes
        const amrex::Real x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

        // Extract particle properties
        const auto ptd = pti.GetParticleTile().getParticleTileData();

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
        const amrex::Real laser_norm = (charge/pc.q_e) * (pc.m_e/mass)
                                     * (charge/pc.q_e) * (pc.m_e/mass);

        int n_qsa_violation = 0;
        amrex::Gpu::DeviceScalar<int> gpu_n_qsa_violation(n_qsa_violation);
        int* const AMREX_RESTRICT p_n_qsa_violation = gpu_n_qsa_violation.dataPtr();

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(isl_fab.box().ixType().cellCentered(),
            "jx, jy, jz, and rho must be cell centered in all directions.");

        Array3<amrex::Real> const isl_arr = isl_fab.array();

        const bool do_tiling = Hipace::m_do_tiling;

        [[maybe_unused]] int ntilex = 1;
        [[maybe_unused]] int ntiley = 1;
        if (do_tiling) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bin_size >= 2*Fields::m_slices_nguards[0],
            "plasmas.sort_bin_size must be at least twice as large as the number of ghost cells");
            amrex::Box domain = gm[lev].Domain();
            domain.coarsen(bin_size);
            ntilex = domain.length(0);
            ntiley = domain.length(1);
        }

#ifndef AMREX_USE_GPU
        for (int tile_perm_x=0; tile_perm_x<2; ++tile_perm_x) {
        for (int tile_perm_y=0; tile_perm_y<2; ++tile_perm_y) {
#ifdef AMREX_USE_OMP
#pragma omp parallel for collapse(2) if(do_tiling)
#endif
        for (int itilex=tile_perm_x; itilex<ntilex; itilex+=2) {
        for (int itiley=tile_perm_y; itiley<ntiley; itiley+=2) {
        // the index is transposed to be the same as in amrex::DenseBins::build
        const int tile_index = itilex * ntiley + itiley;
        using tiling_cto = amrex::CompileTimeOptions<false, true>;
#else
        const int tile_index = 0;
        using tiling_cto = amrex::CompileTimeOptions<false>; // tiling disabled on GPU
#endif

        PlasmaBins::index_type const * const a_indices = do_tiling ? bins.permutationPtr() : nullptr;
        PlasmaBins::index_type const * const a_offsets = do_tiling ? bins.offsetsPtr() : nullptr;

        int num_particles = do_tiling ? a_offsets[tile_index+1]-a_offsets[tile_index]
                                        : pti.numParticles();

        // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
        amrex::ParallelFor(
            amrex::TypeList<
                amrex::CompileTimeOptions<0, 1, 2, 3>,  // depos_order
                amrex::CompileTimeOptions<false, true>, // can_ionize
                tiling_cto,                             // do_tiling
                amrex::CompileTimeOptions<false, true>  // use_laser
            >{}, {
                Hipace::m_depos_order_xy,
                plasma.m_can_ionize,
                do_tiling,
                Hipace::m_use_laser
            },
            num_particles,
            [=] AMREX_GPU_DEVICE (int ip, auto depos_order,
                auto can_ionize, auto c_do_tiling, auto use_laser) noexcept {
            constexpr int depos_order_xy = depos_order.value;

            [[maybe_unused]] auto indices = a_indices;
            [[maybe_unused]] auto offsets = a_offsets;
            [[maybe_unused]] auto itile = tile_index;
            if constexpr (c_do_tiling.value) {
                ip = indices[offsets[itile]+ip];
            }

            // only deposit plasma currents on or below their according MR level
            if (!ptd.id(ip).is_valid() || (lev != 0 && ptd.cpu(ip) < lev)) return;

            const amrex::Real psi_inv = 1._rt/ptd.rdata(PlasmaIdx::psi)[ip];
            const amrex::Real xp = ptd.pos(0, ip);
            const amrex::Real yp = ptd.pos(1, ip);
            const amrex::Real vx_c = ptd.rdata(PlasmaIdx::ux)[ip] * psi_inv;
            const amrex::Real vy_c = ptd.rdata(PlasmaIdx::uy)[ip] * psi_inv;

            // calculate charge of the plasma particles
            amrex::Real q_invvol = charge_invvol * ptd.rdata(PlasmaIdx::w)[ip];
            amrex::Real q_mu0_mass_ratio = charge_mu0_mass_ratio;
            [[maybe_unused]] amrex::Real laser_norm_ion = laser_norm;
            if constexpr (can_ionize.value) {
                q_invvol *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                q_mu0_mass_ratio *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                laser_norm_ion *=
                    ptd.idata(PlasmaIdx::ion_lev)[ip] * ptd.idata(PlasmaIdx::ion_lev)[ip];
            }

            const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
            const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

            amrex::Real Aabssqp = 0._rt;
            [[maybe_unused]] auto a_isl_arr = isl_arr;
            [[maybe_unused]] auto a_aabs = aabs;
            if constexpr (use_laser.value) {
                doLaserGatherShapeN<depos_order_xy>(xp, yp, Aabssqp, a_isl_arr, a_aabs,
                                                    dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                Aabssqp *= laser_norm_ion;
            }

            // calculate gamma/psi for plasma particles
            const amrex::Real gamma_psi = 0.5_rt * (
                (1._rt + 0.5_rt * Aabssqp) * psi_inv * psi_inv
                + vx_c * vx_c * clightinv * clightinv
                + vy_c * vy_c * clightinv * clightinv
                + 1._rt
            );

            if (gamma_psi < 0.0_rt || gamma_psi > max_qsa_weighting_factor || psi_inv < 0.0_rt)
            {
                // This particle violates the QSA, discard it and do not deposit its current
                amrex::Gpu::Atomic::Add(p_n_qsa_violation, 1);
                ptd.rdata(PlasmaIdx::w)[ip] = 0.0_rt;
                ptd.id(ip).make_invalid();
                return;
            }

            for (int iy=0; iy <= depos_order_xy; ++iy) {
                for (int ix=0; ix <= depos_order_xy; ++ix) {

                    // --- Compute shape factors
                    // x direction
                    auto [shape_x, cell_x] =
                        compute_single_shape_factor<false, depos_order_xy>(xmid, ix);

                    // y direction
                    auto [shape_y, cell_y] =
                        compute_single_shape_factor<false, depos_order_xy>(ymid, iy);

                    const amrex::Real charge_density = q_invvol * shape_x * shape_y;
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
        }}}}
#endif

        n_qsa_violation = gpu_n_qsa_violation.dataValue();
        if (n_qsa_violation > 0 && (Hipace::m_verbose >= 3))
            amrex::Print()<< "number of QSA violating particles on this slice: " \
                        << n_qsa_violation << "\n";
    }

    if (deposit_rho && deposit_rho_individual && Hipace::m_deposit_rho) {
        fields.add(lev, which_slice, {"rho"}, which_slice, {rho_str.c_str()});
    }
}
