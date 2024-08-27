/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Severin Diederichs, Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaParticleContainer.H"
#include "utils/Constants.H"
#include "particles/particles_utils/ParticleUtil.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/IonizationEnergiesTable.H"
#include <cmath>

void
PlasmaParticleContainer::
InitParticles (const amrex::RealVect& a_u_std,
               const amrex::RealVect& a_u_mean,
               const amrex::Real a_radius,
               const amrex::Real a_hollow_core_radius)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitParticles()");
    using namespace amrex::literals;
    clearParticles();
    const int lev = 0;
    const auto dx = ParticleGeom(lev).CellSizeArray();
    const auto plo = ParticleGeom(lev).ProbLoArray();
    amrex::RealBox a_bounds = ParticleGeom(lev).ProbDomain();
    a_bounds.setLo(0, Hipace::m_boundary_particle_lo[0]);
    a_bounds.setLo(1, Hipace::m_boundary_particle_lo[1]);
    a_bounds.setHi(0, Hipace::m_boundary_particle_hi[0]);
    a_bounds.setHi(1, Hipace::m_boundary_particle_hi[1]);

    const bool use_fine_patch = m_use_fine_patch;

    const amrex::Array<int, 2> ppc_coarse = m_ppc;
    const int num_ppc_coarse = ppc_coarse[0] * ppc_coarse[1];
    amrex::Real scale_fac_coarse = num_ppc_coarse <= 0 ? 0. :
        (Hipace::m_normalized_units ? 1./num_ppc_coarse : dx[0]*dx[1]*dx[2]/num_ppc_coarse);
    const amrex::Array<int, 2> ppc_fine = m_ppc_fine;
    const int num_ppc_fine = ppc_fine[0] * ppc_fine[1];
    amrex::Real scale_fac_fine = num_ppc_fine <= 0 ? 0. :
        (Hipace::m_normalized_units ? 1./num_ppc_fine : dx[0]*dx[1]*dx[2]/num_ppc_fine);

    amrex::IntVect box_nodal{amrex::IndexType::CELL,amrex::IndexType::CELL,amrex::IndexType::CELL};
    amrex::IntVect box_grow{0, 0, 0};
    amrex::Real x_offset = 0._rt;
    amrex::Real y_offset = 0._rt;

    if (m_prevent_centered_particle) {
        if (ParticleGeom(lev).Domain().length(0) % 2 == 1 && ppc_coarse[0] % 2 == 1) {
            box_nodal[0] = amrex::IndexType::NODE;
            box_grow[0] = -1;
            x_offset = -0.5_rt;
        }

        if (ParticleGeom(lev).Domain().length(1) % 2 == 1 && ppc_coarse[1] % 2 == 1) {
            box_nodal[1] = amrex::IndexType::NODE;
            box_grow[1] = -1;
            y_offset = -0.5_rt;
        }
    }

    for(amrex::MFIter mfi = MakeMFIter(lev, DfltMfi); mfi.isValid(); ++mfi)
    {
        amrex::Box tile_box  = mfi.tilebox(box_nodal, box_grow);

        if (a_radius != std::numeric_limits<amrex::Real>::infinity()) {
            amrex::IntVect lo_limit {
                static_cast<int>(std::round((-a_radius - plo[0])/dx[0] - 2)),
                static_cast<int>(std::round((-a_radius - plo[1])/dx[1] - 2)),
                tile_box.smallEnd(2)
            };
            amrex::IntVect hi_limit {
                static_cast<int>(std::round(( a_radius - plo[0])/dx[0] + 2)),
                static_cast<int>(std::round(( a_radius - plo[1])/dx[1] + 2)),
                tile_box.bigEnd(2)
            };
            tile_box &= amrex::Box(lo_limit, hi_limit, box_nodal);
        }

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        const amrex::Real c_light = get_phys_const().c;
        UpdateDensityFunction(c_light * Hipace::m_physical_time);
        auto density_func = m_density_func;
        const amrex::Real c_t = c_light * Hipace::m_physical_time;
        const amrex::Real min_density = m_min_density;


        amrex::BaseFab<int> fab_fine{};
        if (use_fine_patch) {
            fab_fine.resize(tile_box, 2);
        }
        const Array3<int> arr_fine = fab_fine.array();
        int comp_a = 0;
        int comp_b = 1;
        const int fine_transition_cells = m_fine_transition_cells;
        auto fine_patch_func = m_fine_patch_func;

        if (use_fine_patch) {
            amrex::ParallelFor(to2D(tile_box),
                [=] AMREX_GPU_DEVICE (int i, int j) noexcept
                {
                    const amrex::Real x = plo[0] + (i + 0.5_rt + x_offset)*dx[0];
                    const amrex::Real y = plo[1] + (j + 0.5_rt + y_offset)*dx[1];
                    if (fine_patch_func(x, y) > 0) {
                        arr_fine(i, j, comp_a) = fine_transition_cells + 1;
                    } else {
                        arr_fine(i, j, comp_a) = 0;
                    }
                });

            for (int iter=0; iter<fine_transition_cells; ++iter) {
                std::swap(comp_a, comp_b);

                amrex::ParallelFor(to2D(tile_box),
                    [=] AMREX_GPU_DEVICE (int i, int j) noexcept
                    {
                        int max_val = arr_fine(i, j, comp_b);
                        if (i > lo.x) {
                            max_val = std::max(max_val, arr_fine(i-1, j, comp_b)-1);
                        }
                        if (i < hi.x) {
                            max_val = std::max(max_val, arr_fine(i+1, j, comp_b)-1);
                        }
                        if (j > lo.y) {
                            max_val = std::max(max_val, arr_fine(i, j-1, comp_b)-1);
                        }
                        if (j < hi.y) {
                            max_val = std::max(max_val, arr_fine(i, j+1, comp_b)-1);
                        }
                        arr_fine(i, j, comp_a) = max_val;
                    });
            }
        }

        // Count the total number of particles so only one resize is needed
        amrex::Long total_num_particles = amrex::Reduce::Sum<amrex::Long>(tile_box.numPts(),
            [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
            {
                auto [i,j,k] = tile_box.atOffset3d(idx).arr;

                amrex::Long num_particles_cell = 0;
                for (int i_part=0; i_part<num_ppc_fine; ++i_part)
                {
                    amrex::Real r[2];
                    bool do_init = false;
                    ParticleUtil::get_position_unit_cell_fine(r, do_init, i_part,
                        ppc_coarse, ppc_fine, fine_transition_cells,
                        use_fine_patch ? arr_fine(i, j, comp_a) : 0);

                    if (!do_init) continue;

                    amrex::Real x = plo[0] + (i + r[0] + x_offset)*dx[0];
                    amrex::Real y = plo[1] + (j + r[1] + y_offset)*dx[1];

                    const amrex::Real rsq = x*x + y*y;
                    if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                        y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                        rsq > a_radius*a_radius ||
                        rsq < a_hollow_core_radius*a_hollow_core_radius ||
                        density_func(x, y, c_t) <= min_density) continue;

                    num_particles_cell += 1;
                }
                return num_particles_cell;
            });

        if (m_do_symmetrize) {
            total_num_particles *= 4;
            scale_fac_coarse /= 4.;
            scale_fac_fine /= 4.;
        }

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.size();
        const auto new_size = old_size + total_num_particles;
        particle_tile.resize(new_size);

        auto ptd = particle_tile.getParticleTileData();
        const int init_ion_lev = m_init_ion_lev;

        // The loop over particles is outside the loop over cells
        // so that particles in the same cell are far apart.
        // This makes current deposition faster.
        for (int i_part=0; i_part<num_ppc_fine; ++i_part)
        {
            amrex::Gpu::DeviceVector<unsigned int> counts(tile_box.numPts(), 0);
            unsigned int* pcount = counts.dataPtr();

            amrex::Gpu::DeviceVector<unsigned int> offsets(tile_box.numPts());
            unsigned int* poffset = offsets.dataPtr();

            amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real r[2];
                bool do_init = false;
                ParticleUtil::get_position_unit_cell_fine(r, do_init, i_part,
                    ppc_coarse, ppc_fine, fine_transition_cells,
                    use_fine_patch ? arr_fine(i, j, comp_a) : 0);

                if (!do_init) return;

                amrex::Real x = plo[0] + (i + r[0] + x_offset)*dx[0];
                amrex::Real y = plo[1] + (j + r[1] + y_offset)*dx[1];

                const amrex::Real rsq = x*x + y*y;
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    rsq > a_radius*a_radius ||
                    rsq < a_hollow_core_radius*a_hollow_core_radius ||
                    density_func(x, y, c_t) <= min_density) return;

                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));

                // ordering of axes from fastest to slowest:
                // x
                // y
                // z (not used)
                // ppc
                unsigned int cellid = (uiz * ny + uiy) * nx + uix;

                pcount[cellid] = 1;
            });

            unsigned int num_to_add =
                amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

            if (num_to_add == 0) continue;

            amrex::ParallelForRNG(tile_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, const amrex::RandomEngine& engine) noexcept
            {
                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));

                unsigned int cellid = (uiz * ny + uiy) * nx + uix;

                const amrex::Long pidx = poffset[cellid] - poffset[0] + old_size;

                amrex::Real r[2] = {0.,0.};
                bool do_init = false;
                ParticleUtil::get_position_unit_cell_fine(r, do_init, i_part,
                    ppc_coarse, ppc_fine, fine_transition_cells,
                    use_fine_patch ? arr_fine(i, j, comp_a) : 0);

                if (!do_init) return;

                amrex::Real x = plo[0] + (i + r[0] + x_offset)*dx[0];
                amrex::Real y = plo[1] + (j + r[1] + y_offset)*dx[1];

                const amrex::Real rsq = x*x + y*y;
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    rsq > a_radius*a_radius ||
                    rsq < a_hollow_core_radius*a_hollow_core_radius ||
                    density_func(x, y, c_t) <= min_density) return;

                amrex::Real u[3] = {0.,0.,0.};
                ParticleUtil::get_gaussian_random_momentum(u, a_u_mean, a_u_std, engine);

                ptd.id(pidx) = 1; // plasma id is only used to distinguish between valid/invalid
                ptd.cpu(pidx) = 0; // level 0
                ptd.rdata(PlasmaIdx::x)[pidx] = x;
                ptd.rdata(PlasmaIdx::y)[pidx] = y;

                if (use_fine_patch) {
                    ptd.rdata(PlasmaIdx::w)[pidx] = density_func(x, y, c_t) *
                        (arr_fine(i, j, comp_a) == 0 ? scale_fac_coarse : scale_fac_fine);
                } else {
                    ptd.rdata(PlasmaIdx::w)[pidx] = density_func(x, y, c_t) * scale_fac_coarse;
                }

                ptd.rdata(PlasmaIdx::ux)[pidx] = u[0] * c_light;
                ptd.rdata(PlasmaIdx::uy)[pidx] = u[1] * c_light;
                ptd.rdata(PlasmaIdx::psi)[pidx] = std::sqrt(1._rt+u[0]*u[0]+u[1]*u[1]+u[2]*u[2])-u[2];
                ptd.rdata(PlasmaIdx::x_prev)[pidx] = x;
                ptd.rdata(PlasmaIdx::y_prev)[pidx] = y;
                ptd.rdata(PlasmaIdx::ux_half_step)[pidx] = u[0] * c_light;
                ptd.rdata(PlasmaIdx::uy_half_step)[pidx] = u[1] * c_light;
                ptd.rdata(PlasmaIdx::psi_half_step)[pidx] = ptd.rdata(PlasmaIdx::psi)[pidx];
#ifdef HIPACE_USE_AB5_PUSH
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int iforce = PlasmaIdx::Fx1; iforce <= PlasmaIdx::Fpsi5; ++iforce) {
                    ptd.rdata(iforce)[pidx] = 0._rt;
                }
#endif
                ptd.idata(PlasmaIdx::ion_lev)[pidx] = init_ion_lev;
            });

            old_size += num_to_add;
        }
        if (m_do_symmetrize) {

            const amrex::Real x_mid2 = (a_bounds.lo(0) + a_bounds.hi(0));
            const amrex::Real y_mid2 = (a_bounds.lo(1) + a_bounds.hi(1));
            const amrex::Long mirror_offset = total_num_particles/4;
            amrex::ParallelFor(mirror_offset,
            [=] AMREX_GPU_DEVICE (amrex::Long pidx) noexcept
            {
                const amrex::Real x = ptd.rdata(PlasmaIdx::x)[pidx];
                const amrex::Real y = ptd.rdata(PlasmaIdx::y)[pidx];
                const amrex::Real x_mirror = x_mid2 - x;
                const amrex::Real y_mirror = y_mid2 - y;

                const amrex::Real x_arr[3] = {x_mirror, x, x_mirror};
                const amrex::Real y_arr[3] = {y, y_mirror, y_mirror};
                const amrex::Real ux_arr[3] = {-1._rt, 1._rt, -1._rt};
                const amrex::Real uy_arr[3] = {1._rt, -1._rt, -1._rt};

#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int imirror=0; imirror<3; ++imirror) {
                    const amrex::Long midx = (imirror+1)*mirror_offset +pidx;

                    ptd.id(midx) = 1; // plasma id is only used to distinguish between valid/invalid
                    ptd.cpu(midx) = 0; // level 0
                    ptd.rdata(PlasmaIdx::x)[midx] = x_arr[imirror];
                    ptd.rdata(PlasmaIdx::y)[midx] = y_arr[imirror];

                    ptd.rdata(PlasmaIdx::w)[midx] = ptd.rdata(PlasmaIdx::w)[pidx];
                    ptd.rdata(PlasmaIdx::ux)[midx] = ptd.rdata(PlasmaIdx::ux)[pidx] * ux_arr[imirror];
                    ptd.rdata(PlasmaIdx::uy)[midx] = ptd.rdata(PlasmaIdx::uy)[pidx] * uy_arr[imirror];
                    ptd.rdata(PlasmaIdx::psi)[midx] = ptd.rdata(PlasmaIdx::psi)[pidx];
                    ptd.rdata(PlasmaIdx::x_prev)[midx] = x_arr[imirror];
                    ptd.rdata(PlasmaIdx::y_prev)[midx] = y_arr[imirror];
                    ptd.rdata(PlasmaIdx::ux_half_step)[midx] =
                        ptd.rdata(PlasmaIdx::ux_half_step)[pidx] * ux_arr[imirror];
                    ptd.rdata(PlasmaIdx::uy_half_step)[midx] =
                        ptd.rdata(PlasmaIdx::uy_half_step)[pidx] * uy_arr[imirror];
                    ptd.rdata(PlasmaIdx::psi_half_step)[midx] =
                        ptd.rdata(PlasmaIdx::psi_half_step)[pidx];
#ifdef HIPACE_USE_AB5_PUSH
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                    for (int iforce = PlasmaIdx::Fx1; iforce <= PlasmaIdx::Fpsi5; ++iforce) {
                        ptd.rdata(iforce)[midx] = 0._rt;
                    }
#endif
                    ptd.idata(PlasmaIdx::ion_lev)[midx] = ptd.idata(PlasmaIdx::ion_lev)[pidx];

                }
            });
        }
    }
#ifndef AMREX_USE_GPU
    // The index order for initializing plasma particles is optimized for GPU.
    // On CPU, especially with multiple particles per cell,
    // reordering particles by cell gives better performance for plasma deposition and push.
    SortParticlesByCell();
#endif
}

void
PlasmaParticleContainer::
InitIonizationModule (const amrex::Geometry& geom, PlasmaParticleContainer* product_pc,
                      const amrex::Real background_density_SI)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitIonizationModule()");

    using namespace amrex::literals;

    if (!m_can_ionize) return;

    const bool normalized_units = Hipace::m_normalized_units;
    if (normalized_units) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(background_density_SI!=0,
            "For ionization with normalized units, a background plasma density != 0 must "
            "be specified via 'hipace.background_density_SI'");
    }

    m_product_pc = product_pc;
    amrex::ParmParse pp(m_name);
    std::string physical_element;
    getWithParser(pp, "element", physical_element);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ion_map_ids.count(physical_element) != 0,
        "There are no ionization energies available for this element. "
        "Please update src/utils/IonizationEnergiesTable.H using write_atomic_data_cpp.py");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE((std::abs(product_pc->m_charge / m_charge +1) < 1e-3),
        "Ion and Ionization product charges have to be opposite");
    // Get atomic number and ionization energies from file
    const int ion_element_id = ion_map_ids[physical_element];
    const int ion_atomic_number = ion_atomic_numbers[ion_element_id];
    amrex::Vector<amrex::Real> h_ionization_energies(ion_atomic_number);
    const int offset = ion_energy_offsets[ion_element_id];
    for(int i=0; i<ion_atomic_number; i++){
        h_ionization_energies[i] = table_ionization_energies[i+offset];
    }
    // Compute ADK prefactors (See Chen, JCP 236 (2013), equation (2))
    // For now, we assume l=0 and m=0.
    // The approximate expressions are used,
    // without Gamma function
    const PhysConst phys_const = make_constants_SI();
    const amrex::Real alpha = 0.0072973525693_rt;
    const amrex::Real r_e = 2.8179403227e-15_rt;
    const amrex::Real a3 = alpha * alpha * alpha;
    const amrex::Real a4 = a3 * alpha;
    const amrex::Real wa = a3 * phys_const.c / r_e;
    const amrex::Real Ea = phys_const.m_e * phys_const.c * phys_const.c / phys_const.q_e * a4 / r_e;
    const amrex::Real UH = table_ionization_energies[0];
    const amrex::Real l_eff = std::sqrt(UH/h_ionization_energies[0]) - 1._rt;

    // plasma frequency in SI units to denormalize ionization
    const amrex::Real wp = std::sqrt(static_cast<double>(background_density_SI) *
                                     PhysConstSI::q_e*PhysConstSI::q_e /
                                     (PhysConstSI::ep0 * PhysConstSI::m_e) );
    const amrex::Real dt = normalized_units ? geom.CellSize(2)/wp : geom.CellSize(2)/phys_const.c;

    m_adk_power.resize(ion_atomic_number);
    m_adk_prefactor.resize(ion_atomic_number);
    m_adk_exp_prefactor.resize(ion_atomic_number);

    amrex::Gpu::PinnedVector<amrex::Real> h_adk_power(ion_atomic_number);
    amrex::Gpu::PinnedVector<amrex::Real> h_adk_prefactor(ion_atomic_number);
    amrex::Gpu::PinnedVector<amrex::Real> h_adk_exp_prefactor(ion_atomic_number);

    for (int i=0; i<ion_atomic_number; ++i)
    {
        const amrex::Real n_eff = (i+1) * std::sqrt(UH/h_ionization_energies[i]);
        const amrex::Real C2 = std::pow(2,2*n_eff)/(n_eff*std::tgamma(n_eff+l_eff+1)
                         * std::tgamma(n_eff-l_eff));
        h_adk_power[i] = -(2*n_eff - 1);
        const amrex::Real Uion = h_ionization_energies[i];
        h_adk_prefactor[i] = dt * wa * C2 * ( Uion/(2*UH) )
            * std::pow(2*std::pow((Uion/UH),3./2)*Ea,2*n_eff - 1);
        h_adk_exp_prefactor[i] = -2./3 * std::pow( Uion/UH,3./2) * Ea;
    }

    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
        h_adk_power.begin(), h_adk_power.end(), m_adk_power.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
        h_adk_prefactor.begin(), h_adk_prefactor.end(), m_adk_prefactor.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
        h_adk_exp_prefactor.begin(), h_adk_exp_prefactor.end(), m_adk_exp_prefactor.begin());
}
