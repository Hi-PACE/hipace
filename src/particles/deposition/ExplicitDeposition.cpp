/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "ExplicitDeposition.H"

#include "Hipace.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"

#include "AMReX_GpuLaunch.H"


struct GetCellRetrun {
    int x;
    int y;
    bool valid;
};

template<int stencil_x, int stencil_y, class L, class F, std::size_t max_depos, std::size_t max_cache>
void
SharedMemoryDeposition (int num_particles, L&& get_start_cell, F&& do_deposit,
                        Array3<amrex::Real> field, amrex::Box box,
                        std::array<int, max_cache> idx_cache,
                        std::array<int, max_depos> idx_depos) {
    constexpr int threads_per_tile = 256;
    constexpr int tile_x = 16;
    constexpr int tile_y = 16;
    constexpr int tile_s_x = tile_x + stencil_x - 1;
    constexpr int tile_s_y = tile_y + stencil_y - 1;

    const int lo_x = box.smallEnd(0);
    const int lo_y = box.smallEnd(1);
    const int hi_x = box.bigEnd(0);
    const int hi_y = box.bigEnd(1);
    const int ntile_x = (box.length(0) + tile_x - 1) / tile_x;
    const int ntile_y = (box.length(1) + tile_y - 1) / tile_y;
    constexpr int ll_guard = std::numeric_limits<int>::max();
    amrex::Gpu::DeviceVector<int> ll_start(ntile_x * ntile_y * threads_per_tile, ll_guard);
    amrex::Gpu::DeviceVector<int> ll_count(ntile_x * ntile_y, 0);
    amrex::Gpu::DeviceVector<int> ll_next(num_particles);
    int * const p_ll_start = ll_start.dataPtr();
    int * const p_ll_count = ll_count.dataPtr();
    int * const p_ll_next = ll_next.dataPtr();

    {
        HIPACE_PROFILE("test1()");

    amrex::ParallelFor(num_particles,
        [=] AMREX_GPU_DEVICE (int ip) {
            ip = num_particles - ip - 1;
            auto [cell_x, cell_y, is_valid] = get_start_cell(ip);
            if (is_valid) {
                const int tile_id_x = (cell_x - lo_x) / tile_x;
                const int tile_id_y = (cell_y - lo_y) / tile_y;
                const int idx = (tile_id_x + tile_id_y * ntile_x);

                //const int count = amrex::Gpu::Atomic::Add(p_ll_count + idx, 1);

                //p_ll_next[ip] = amrex::Gpu::Atomic::Exch(
                //    p_ll_start + idx*threads_per_tile + count % threads_per_tile, ip);

                const int loc_id_x = (cell_x - lo_x - tile_id_x * tile_x);
                const int loc_id_y = (cell_y - lo_y - tile_id_y * tile_y);
                const int loc_id = loc_id_x + loc_id_y * tile_x;

                p_ll_next[ip] = amrex::Gpu::Atomic::Exch(
                    p_ll_start + idx*threads_per_tile + loc_id, ip);
            }
        });

    }

    /*{
        HIPACE_PROFILE("test0()");
        amrex::Gpu::DeviceVector<int> ll_start_2(ll_start.size());
        int * const p_ll_start_2 = ll_start_2.dataPtr();
        amrex::ParallelFor(ll_start.size(),
            [=] AMREX_GPU_DEVICE (int ip) {
                p_ll_start_2[ip] = p_ll_start[ip];
            });

        std::cout << "depos";
        int num_par = 0;
        do {
            num_par = amrex::Reduce::Sum(ll_start.size(), [=]AMREX_GPU_DEVICE(int ip){
                int idx = p_ll_start_2[ip];
                if (idx != ll_guard) {
                    p_ll_start_2[ip] = p_ll_next[idx];
                    return 1;
                } else {
                    return 0;
                }
            }, 0);
            std::cout << " " << num_par;
        } while (num_par > 0);
        std::cout << std::endl;
    }*/

    amrex::Math::FastDivmodU64 num_tiles_divmod {static_cast<std::uint64_t>(ntile_x)};

    HIPACE_PROFILE("test2()");

    const int niter = Hipace::m_shared_depos_iter;

    amrex::launch<threads_per_tile>(ntile_x * ntile_y, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE () {

            __shared__ amrex::Real shared_ptr[tile_s_x * tile_s_y * (max_cache + max_depos)];

            const int tile_id = blockIdx.x;

            std::uint64_t remainder = 0;
            const int tile_id_y = num_tiles_divmod.divmod(remainder, tile_id);
            const int tile_id_x = remainder;

            const int tile_begin_x = lo_x + tile_id_x * tile_x;
            const int tile_begin_y = lo_y + tile_id_y * tile_y;

            const int tile_end_x = std::min(tile_begin_x + tile_s_x, hi_x + 1);
            const int tile_end_y = std::min(tile_begin_y + tile_s_y, hi_y + 1);

            Array3<amrex::Real> shared_arr{{
                shared_ptr,
                {tile_begin_x, tile_begin_y, 0},
                {tile_end_x, tile_end_y, 1},
                max_cache + max_depos
            }};

            for (int s = threadIdx.x; s < tile_s_x * tile_s_y; s+=threads_per_tile) {
                int sy = s / tile_s_x;
                int sx = s - sy * tile_s_x;
                sx += tile_begin_x;
                sy += tile_begin_y;
                if (sx <= hi_x && sy <= hi_y) {
                    for (int n=0; n != max_cache; ++n) {
                        if (idx_cache[n] != -1) {
                            shared_arr(sx, sy, n) = field(sx, sy, idx_cache[n]);
                        }
                    }
                    for (int n=0; n != max_depos; ++n) {
                        if (idx_depos[n] != -1) {
                            shared_arr(sx, sy, n+max_cache) = 0;
                        }
                    }
                }
            }

            __syncthreads();

            int current_idx = p_ll_start[tile_id * threads_per_tile + threadIdx.x];

            while (current_idx != ll_guard) {

                do_deposit(current_idx, shared_arr, field);

                current_idx = p_ll_next[current_idx];
            }

            __syncthreads();

            for (int s = threadIdx.x; s < tile_s_x * tile_s_y; s+=threads_per_tile) {
                int sy = s / tile_s_x;
                int sx = s - sy * tile_s_x;
                sx += tile_begin_x;
                sy += tile_begin_y;
                if (sx <= hi_x && sy <= hi_y) {
                    for (int n=0; n != max_depos; ++n) {
                        if (idx_depos[n] != -1) {
                            amrex::Gpu::Atomic::Add(
                                field.ptr(sx, sy, idx_depos[n]), shared_arr(sx, sy, n+max_cache));
                        }
                    }
                }
            }


        }
    );

    amrex::Gpu::streamSynchronize();
}




void
ExplicitDepositionS (PlasmaParticleContainer& plasma, Fields& fields,
                    amrex::Vector<amrex::Geometry> const& gm, const int lev) {
    HIPACE_PROFILE("ExplicitDeposition()");
    using namespace amrex::literals;

    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {

        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];
        const Array3<amrex::Real> field_arr = isl_fab.array();

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];

        const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
        const int EypBx = Comps[WhichSlice::This]["EypBx"];
        const int Ez = Comps[WhichSlice::This]["Ez"];
        const int Bz = Comps[WhichSlice::This]["Bz"];
        const int aabs_comp = Hipace::m_use_laser ? Comps[WhichSlice::This]["aabs"] : -1;

        // Extract particle properties
        const auto ptd = pti.GetParticleTile().getParticleTileData();

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
        const amrex::Real charge_invvol_mu0 = plasma.m_charge * invvol * pc.mu0 / Hipace::m_shared_depos_iter;
        const amrex::Real charge_mass_ratio = plasma.m_charge / plasma.m_mass;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        {

        amrex::Long const num_particles = pti.numParticles();
#ifdef AMREX_USE_OMP
        amrex::Long const idx_begin = (num_particles * omp_get_thread_num()) / omp_get_num_threads();
        amrex::Long const idx_end = (num_particles * (omp_get_thread_num()+1)) / omp_get_num_threads();
        const bool do_omp_atomic = omp_get_num_threads() > 1;
        using omp_cto = amrex::CompileTimeOptions<false, true>;
#else
        amrex::Long constexpr idx_begin = 0;
        amrex::Long const idx_end = num_particles;
        constexpr bool do_omp_atomic = false;
        using omp_cto = amrex::CompileTimeOptions<false>;
#endif

        auto get_cell = [=] AMREX_GPU_DEVICE (int ip,
                auto depos_order, auto derivative_type, auto can_ionize, auto use_laser, auto ompa)
                -> GetCellRetrun {
            if (!ptd.id(ip).is_valid() || (lev != 0 && ptd.cpu(ip) < lev)) return {0, 0, false};

            const amrex::Real xp = ptd.pos(0, ip);
            const amrex::Real yp = ptd.pos(1, ip);

            const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
            const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

            auto [shape_y, shape_dy, j] =
                single_derivative_shape_factor<derivative_type, depos_order>(ymid, 0);
            auto [shape_x, shape_dx, i] =
                single_derivative_shape_factor<derivative_type, depos_order>(xmid, 0);

            return {i, j, true};
        };

        amrex::AnyCTO(
            amrex::TypeList<
                amrex::CompileTimeOptions<0, 1, 2, 3>,  // depos_order
                amrex::CompileTimeOptions<0, 1, 2>,     // derivative_type
                amrex::CompileTimeOptions<false, true>, // can_ionize
                amrex::CompileTimeOptions<false, true>, // use_laser
                omp_cto                                 // do_omp_atomic
            >{}, {
                Hipace::m_depos_order_xy,
                Hipace::m_depos_derivative_type,
                plasma.m_can_ionize,
                Hipace::m_use_laser,
                do_omp_atomic
            },
            [&](auto cto_func){
                constexpr auto ctos = cto_func.GetOptions();
                constexpr int c_depos_order = ctos[0];
                constexpr int c_derivative_type = ctos[1];
                constexpr int stencil_size = c_depos_order + c_derivative_type + 1;
                SharedMemoryDeposition<stencil_size, stencil_size>(
                    int(idx_end - idx_begin), cto_func.Wrap(get_cell), cto_func, field_arr,
                    isl_fab.box(), std::array{Bz, Ez, ExmBy, EypBx}, std::array{Sy, Sx});
            },
            [=] AMREX_GPU_DEVICE (int ip, Array3<amrex::Real> larr, Array3<amrex::Real> garr,
                                  auto depos_order, auto derivative_type,
                                  auto can_ionize, auto use_laser, auto ompa) noexcept {
                ip += idx_begin;

                if (!ptd.id(ip).is_valid() || (lev != 0 && ptd.cpu(ip) < lev)) return;
                const amrex::Real psi_inv = 1._rt/ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);
                const amrex::Real vx = ptd.rdata(PlasmaIdx::ux)[ip] * psi_inv * clight_inv;
                const amrex::Real vy = ptd.rdata(PlasmaIdx::uy)[ip] * psi_inv * clight_inv;

                // Rename variable for NVCC lambda capture to work
                amrex::Real q_invvol_mu0 = charge_invvol_mu0;
                amrex::Real q_mass_ratio = charge_mass_ratio;
                if constexpr (can_ionize.value) {
                    q_invvol_mu0 *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                    q_mass_ratio *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                }

                const amrex::Real charge_density_mu0 = q_invvol_mu0 * ptd.rdata(PlasmaIdx::w)[ip];

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                amrex::Real Aabssqp = 0._rt;
                if (use_laser.value) {
                    // Its important that Aabssqp is first fully gathered and not used
                    // directly per cell like AabssqDxp and AabssqDyp
                    doLaserGatherShapeN<depos_order>(xp, yp, Aabssqp, garr, aabs_comp,
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
                        const amrex::Real Bz_v = larr(i,j,0);
                        const amrex::Real Ez_v = larr(i,j,1);
                        const amrex::Real ExmBy_v = larr(i,j,2);
                        const amrex::Real EypBx_v = larr(i,j,3);

                        amrex::Real AabssqDxp = 0._rt;
                        amrex::Real AabssqDyp = 0._rt;
                        // Rename variables for NVCC lambda capture to work
                        [[maybe_unused]] auto clight = a_clight;
                        if constexpr (use_laser.value) {
                            // avoid going outside of domain
                            if (shape_x * shape_y != 0._rt) {
                                const amrex::Real xp1y00 = garr(i+1, j  , aabs_comp);
                                const amrex::Real xm1y00 = garr(i-1, j  , aabs_comp);
                                const amrex::Real x00yp1 = garr(i  , j+1, aabs_comp);
                                const amrex::Real x00ym1 = garr(i  , j-1, aabs_comp);
                                AabssqDxp = (xp1y00-xm1y00) * 0.5_rt * dx_inv * laser_fac * clight;
                                AabssqDyp = (x00yp1-x00ym1) * 0.5_rt * dy_inv * laser_fac * clight;
                            }
                        }

                        AtomicAdd(larr.ptr(i, j, 4), charge_density_mu0 * (
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
                        ), ompa.value);

                        AtomicAdd(larr.ptr(i, j, 5), charge_density_mu0 * (
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
                        ), ompa.value);
                    }
                }
            });
        }
    }
}







void
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields,
                    amrex::Vector<amrex::Geometry> const& gm, const int lev) {
    HIPACE_PROFILE("ExplicitDeposition()");
    using namespace amrex::literals;

    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {

        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];
        const Array3<amrex::Real> arr = isl_fab.array();

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];

        const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
        const int EypBx = Comps[WhichSlice::This]["EypBx"];
        const int Ez = Comps[WhichSlice::This]["Ez"];
        const int Bz = Comps[WhichSlice::This]["Bz"];
        const int aabs_comp = Hipace::m_use_laser ? Comps[WhichSlice::This]["aabs"] : -1;

        // Extract particle properties
        const auto ptd = pti.GetParticleTile().getParticleTileData();

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

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        {

        amrex::Long const num_particles = pti.numParticles();
#ifdef AMREX_USE_OMP
        amrex::Long const idx_begin = (num_particles * omp_get_thread_num()) / omp_get_num_threads();
        amrex::Long const idx_end = (num_particles * (omp_get_thread_num()+1)) / omp_get_num_threads();
        const bool do_omp_atomic = omp_get_num_threads() > 1;
        using omp_cto = amrex::CompileTimeOptions<false, true>;
#else
        amrex::Long constexpr idx_begin = 0;
        amrex::Long const idx_end = num_particles;
        constexpr bool do_omp_atomic = false;
        using omp_cto = amrex::CompileTimeOptions<false>;
#endif

        amrex::ParallelFor(
            amrex::TypeList<
                amrex::CompileTimeOptions<0, 1, 2, 3>,  // depos_order
                amrex::CompileTimeOptions<0, 1, 2>,     // derivative_type
                amrex::CompileTimeOptions<false, true>, // can_ionize
                amrex::CompileTimeOptions<false, true>, // use_laser
                omp_cto                                 // do_omp_atomic
            >{}, {
                Hipace::m_depos_order_xy,
                Hipace::m_depos_derivative_type,
                plasma.m_can_ionize,
                Hipace::m_use_laser,
                do_omp_atomic
            },
            int(idx_end - idx_begin), // int ParallelFor is 3-5% faster than amrex::Long version
            [=] AMREX_GPU_DEVICE (int ip, auto a_depos_order, auto a_derivative_type,
                                  auto can_ionize, auto use_laser, auto ompa) noexcept {
                constexpr int depos_order = a_depos_order.value;
                constexpr int derivative_type = a_derivative_type.value;

                ip += idx_begin;

                if (!ptd.id(ip).is_valid() || (lev != 0 && ptd.cpu(ip) < lev)) return;
                const amrex::Real psi_inv = 1._rt/ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);
                const amrex::Real vx = ptd.rdata(PlasmaIdx::ux)[ip] * psi_inv * clight_inv;
                const amrex::Real vy = ptd.rdata(PlasmaIdx::uy)[ip] * psi_inv * clight_inv;

                // Rename variable for NVCC lambda capture to work
                amrex::Real q_invvol_mu0 = charge_invvol_mu0;
                amrex::Real q_mass_ratio = charge_mass_ratio;
                if constexpr (can_ionize.value) {
                    q_invvol_mu0 *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                    q_mass_ratio *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                }

                const amrex::Real charge_density_mu0 = q_invvol_mu0 * ptd.rdata(PlasmaIdx::w)[ip];

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                amrex::Real Aabssqp = 0._rt;
                if (use_laser.value) {
                    // Its important that Aabssqp is first fully gathered and not used
                    // directly per cell like AabssqDxp and AabssqDyp
                    doLaserGatherShapeN<depos_order>(xp, yp, Aabssqp, arr, aabs_comp,
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
                        const amrex::Real Bz_v = arr(i,j,Bz);
                        const amrex::Real Ez_v = arr(i,j,Ez);
                        const amrex::Real ExmBy_v = arr(i,j,ExmBy);
                        const amrex::Real EypBx_v = arr(i,j,EypBx);

                        amrex::Real AabssqDxp = 0._rt;
                        amrex::Real AabssqDyp = 0._rt;
                        // Rename variables for NVCC lambda capture to work
                        [[maybe_unused]] auto clight = a_clight;
                        if constexpr (use_laser.value) {
                            // avoid going outside of domain
                            if (shape_x * shape_y != 0._rt) {
                                const amrex::Real xp1y00 = arr(i+1, j  , aabs_comp);
                                const amrex::Real xm1y00 = arr(i-1, j  , aabs_comp);
                                const amrex::Real x00yp1 = arr(i  , j+1, aabs_comp);
                                const amrex::Real x00ym1 = arr(i  , j-1, aabs_comp);
                                AabssqDxp = (xp1y00-xm1y00) * 0.5_rt * dx_inv * laser_fac * clight;
                                AabssqDyp = (x00yp1-x00ym1) * 0.5_rt * dy_inv * laser_fac * clight;
                            }
                        }

                        AtomicAdd(arr.ptr(i, j, Sy), charge_density_mu0 * (
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
                        ), ompa.value);

                        AtomicAdd(arr.ptr(i, j, Sx), charge_density_mu0 * (
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
                        ), ompa.value);
                    }
                }
            });
        }
    }
}
