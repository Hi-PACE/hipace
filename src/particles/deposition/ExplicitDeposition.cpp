/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "ExplicitDeposition.H"

#include "DepositionUtil.H"
#include "Hipace.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"

#include "AMReX_GpuLaunch.H"

void
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields,
                    amrex::Vector<amrex::Geometry> const& gm, const int lev) {
    HIPACE_PROFILE("ExplicitDeposition()");
    using namespace amrex::literals;

    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {

        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];

        const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
        const int EypBx = Comps[WhichSlice::This]["EypBx"];
        const int Ez = Comps[WhichSlice::This]["Ez"];
        const int Bz = Comps[WhichSlice::This]["Bz"];
        const int aabs_comp = Hipace::m_use_laser ? Comps[WhichSlice::This]["aabs"] : -1;

        const amrex::Real x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

        const amrex::Real dx_inv = gm[lev].InvCellSize(0);
        const amrex::Real dy_inv = gm[lev].InvCellSize(1);
        const amrex::Real dz_inv = gm[lev].InvCellSize(2);
        // in normalized units this is rescaling dx and dy for MR,
        // while in SI units it's the factor for charge to charge density
        const amrex::Real invvol = Hipace::m_normalized_units ?
            gm[0].CellSize(0)*gm[0].CellSize(1)*dx_inv*dy_inv
            : dx_inv*dy_inv*dz_inv;

        const PhysConst pc = get_phys_const();
        const amrex::Real a_clight = pc.c;
        const amrex::Real clight_inv = 1._rt/pc.c;
        // The laser a0 is always normalized
        const amrex::Real laser_fac = (pc.m_e/pc.q_e) * (pc.m_e/pc.q_e);
        const amrex::Real charge_invvol_mu0 = plasma.m_charge * invvol * pc.mu0;
        const amrex::Real charge_mass_ratio = plasma.m_charge / plasma.m_mass;

        amrex::AnyCTO(
            // use compile-time options
            amrex::TypeList<
                amrex::CompileTimeOptions<0, 1, 2, 3>,  // depos_order
                amrex::CompileTimeOptions<0, 1, 2>,     // derivative_type
                amrex::CompileTimeOptions<false, true>, // can_ionize
                amrex::CompileTimeOptions<false, true>  // use_laser
            >{}, {
                Hipace::m_depos_order_xy,
                Hipace::m_depos_derivative_type,
                plasma.m_can_ionize,
                Hipace::m_use_laser
            },
            // call deposition function
            // The three functions passed as arguments to this lambda
            // are defined below as the next arguments.
            [&](auto is_valid, auto get_cell, auto deposit){
                constexpr auto ctos = deposit.GetOptions();
                constexpr int depos_order = ctos[0];
                constexpr int derivative_type = ctos[1];
                constexpr int use_laser = ctos[3];
                if constexpr (use_laser) {
                    // need extra cells for gathering the laser
                    constexpr int stencil_size = depos_order + 2 + 1;
                    SharedMemoryDeposition<stencil_size, stencil_size, false>(
                        int(pti.numParticles()), is_valid, get_cell, deposit, isl_fab.array(),
                        isl_fab.box(), pti.GetParticleTile().getParticleTileData(),
                        std::array{Bz, Ez, ExmBy, EypBx, aabs_comp}, std::array{Sy, Sx});
                } else {
                    constexpr int stencil_size = depos_order + derivative_type + 1;
                    SharedMemoryDeposition<stencil_size, stencil_size, false>(
                        int(pti.numParticles()), is_valid, get_cell, deposit, isl_fab.array(),
                        isl_fab.box(), pti.GetParticleTile().getParticleTileData(),
                        std::array{Bz, Ez, ExmBy, EypBx}, std::array{Sy, Sx});
                }
            },
            // is_valid
            // return whether the particle is valid and should deposit
            [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                                  auto /*depos_order*/,
                                  auto /*derivative_type*/,
                                  auto /*can_ionize*/,
                                  auto /*use_laser*/)
            {
                // only deposit plasma Sx and Sy on or below their according MR level
                return ptd.id(ip).is_valid() && (lev == 0 || ptd.cpu(ip) >= lev);
            },
            // get_cell
            // return the lowest cell index that the particle deposits into
            [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                                  auto depos_order,
                                  auto derivative_type,
                                  auto /*can_ionize*/,
                                  auto use_laser) -> amrex::IntVectND<2>
            {
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                auto [shape_y, shape_dy, j] =
                    single_derivative_shape_factor<derivative_type, depos_order>(ymid, 0);
                auto [shape_x, shape_dx, i] =
                    single_derivative_shape_factor<derivative_type, depos_order>(xmid, 0);

                if constexpr (use_laser) {
                    // need extra cells for gathering the laser
                    if constexpr (derivative_type == 0) {
                        return {i-1, j-1};
                    } else if constexpr (derivative_type == 1) {
                        return {shape_x == 0._rt ? i : i-1, shape_y == 0._rt ? j : j-1};
                    } else {
                        return {i, j};
                    }
                } else {
                    return {i, j};
                }
            },
            // deposit
            // deposit the charge / current of one particle
            [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                                  Array3<amrex::Real> arr,
                                  auto cache_idx, auto depos_idx,
                                  auto depos_order,
                                  auto derivative_type,
                                  auto can_ionize,
                                  auto use_laser) noexcept
            {
                const amrex::Real psi_inv = 1._rt/ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);
                const amrex::Real vx = ptd.rdata(PlasmaIdx::ux)[ip] * psi_inv * clight_inv;
                const amrex::Real vy = ptd.rdata(PlasmaIdx::uy)[ip] * psi_inv * clight_inv;

                // Rename variable for NVCC lambda capture to work
                amrex::Real q_invvol_mu0 = charge_invvol_mu0;
                amrex::Real q_mass_ratio = charge_mass_ratio;
                if constexpr (can_ionize) {
                    q_invvol_mu0 *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                    q_mass_ratio *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                }

                const amrex::Real charge_density_mu0 = q_invvol_mu0 * ptd.rdata(PlasmaIdx::w)[ip];

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                amrex::Real Aabssqp = 0._rt;
                if (use_laser) {
                    // Its important that Aabssqp is first fully gathered and not used
                    // directly per cell like AabssqDxp and AabssqDyp
                    doLaserGatherShapeN<depos_order>(xp, yp, Aabssqp, arr, cache_idx[4],
                                                     dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                    Aabssqp *= laser_fac * q_mass_ratio * q_mass_ratio;
                }

                // calculate gamma/psi for plasma particles
                const amrex::Real gamma_psi = 0.5_rt * (
                    (1._rt + 0.5_rt * Aabssqp) * psi_inv * psi_inv
                    + vx * vx
                    + vy * vy
                    + 1._rt
                );

#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int iy=0; iy <= depos_order+derivative_type; ++iy) {
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                    for (int ix=0; ix <= depos_order+derivative_type; ++ix) {

                        if constexpr (derivative_type == 2) {
                            if ((ix==0 || ix==depos_order + 2) && (iy==0 || iy==depos_order + 2)) {
                                // corners have a shape factor of zero
                                continue;
                            }
                        }

                        auto [shape_y, shape_dy, j] =
                            single_derivative_shape_factor<derivative_type, depos_order>(ymid, iy);
                        auto [shape_x, shape_dx, i] =
                            single_derivative_shape_factor<derivative_type, depos_order>(xmid, ix);

                        // get fields per cell instead of gathering them to avoid blurring
                        const amrex::Real Bz_v = arr(i, j, cache_idx[0]);
                        const amrex::Real Ez_v = arr(i, j, cache_idx[1]);
                        const amrex::Real ExmBy_v = arr(i, j, cache_idx[2]);
                        const amrex::Real EypBx_v = arr(i, j, cache_idx[3]);

                        amrex::Real AabssqDxp = 0._rt;
                        amrex::Real AabssqDyp = 0._rt;
                        // Rename variables for NVCC lambda capture to work
                        [[maybe_unused]] auto clight = a_clight;
                        if constexpr (use_laser) {
                            // avoid going outside of domain
                            if (shape_x * shape_y != 0._rt) {
                                // need extra cells for gathering the laser
                                const amrex::Real xp1y00 = arr(i+1, j  , cache_idx[4]);
                                const amrex::Real xm1y00 = arr(i-1, j  , cache_idx[4]);
                                const amrex::Real x00yp1 = arr(i  , j+1, cache_idx[4]);
                                const amrex::Real x00ym1 = arr(i  , j-1, cache_idx[4]);
                                AabssqDxp = (xp1y00-xm1y00) * 0.5_rt * dx_inv * laser_fac * clight;
                                AabssqDyp = (x00yp1-x00ym1) * 0.5_rt * dy_inv * laser_fac * clight;
                            }
                        }

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[0]), charge_density_mu0 * (
                            - shape_x * shape_y * (
                                - Bz_v * vx
                                + ( Ez_v * vy
                                + ExmBy_v * (          - vx * vy)
                                + EypBx_v * (gamma_psi - vy * vy) ) * clight_inv
                                - 0.25_rt * AabssqDyp * q_mass_ratio * psi_inv
                            ) * q_mass_ratio * psi_inv
                            + ( - shape_dx * shape_y * dx_inv * (
                                - vx * vy
                            )
                            - shape_x * shape_dy * dy_inv * (
                                gamma_psi - vy * vy - 1._rt
                            )) * a_clight
                        ));

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[1]), charge_density_mu0 * (
                            + shape_x * shape_y * (
                                + Bz_v * vy
                                + ( Ez_v * vx
                                + ExmBy_v * (gamma_psi - vx * vx)
                                + EypBx_v * (          - vx * vy) ) * clight_inv
                                - 0.25_rt * AabssqDxp * q_mass_ratio * psi_inv
                            ) * q_mass_ratio * psi_inv
                            + ( + shape_dx * shape_y * dx_inv * (
                                gamma_psi - vx * vx - 1._rt
                            )
                            + shape_x * shape_dy * dy_inv * (
                                - vx * vy
                            )) * a_clight
                        ));
                    }
                }
            });
    }
}
