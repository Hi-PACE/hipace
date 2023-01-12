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
InitParticles (const amrex::IntVect& a_num_particles_per_cell,
               const amrex::RealVect& a_u_std,
               const amrex::RealVect& a_u_mean,
               const amrex::Real a_radius,
               const amrex::Real a_hollow_core_radius)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitParticles");
    using namespace amrex::literals;
    const int lev = m_level;
    const auto dx = ParticleGeom(lev).CellSizeArray();
    const auto plo = ParticleGeom(lev).ProbLoArray();
    const amrex::RealBox a_bounds = ParticleGeom(lev).ProbDomain();

    const int depos_order_1 = Hipace::m_depos_order_xy + 1;
    const bool outer_depos_loop = Hipace::m_outer_depos_loop;

    const int num_ppc = AMREX_D_TERM( a_num_particles_per_cell[0],
                                      *a_num_particles_per_cell[1],
                                      *a_num_particles_per_cell[2]);
    const amrex::Real scale_fac = Hipace::m_normalized_units?
                                  1./num_ppc : dx[0]*dx[1]*dx[2]/num_ppc;

    for(amrex::MFIter mfi = MakeMFIter(lev, DfltMfi); mfi.isValid(); ++mfi)
    {

        const amrex::Box& tile_box  = mfi.tilebox();

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        amrex::Gpu::DeviceVector<unsigned int> counts(tile_box.numPts()*num_ppc, 0);
        unsigned int* pcount = counts.dataPtr();

        amrex::Gpu::DeviceVector<unsigned int> offsets(tile_box.numPts()*num_ppc);
        unsigned int* poffset = offsets.dataPtr();

        UpdateDensityFunction();
        auto density_func = m_density_func;
        const amrex::Real c_light = get_phys_const().c;
        const amrex::Real c_t = c_light * Hipace::m_physical_time;
        const amrex::Real min_density = m_min_density;

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3];

                ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];

                const amrex::Real rsq = x*x + y*y;
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    rsq > a_radius*a_radius ||
                    rsq < a_hollow_core_radius*a_hollow_core_radius ||
                    density_func(x, y, c_t) < min_density) continue;

                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));

                unsigned int cellid = 0;
                if (outer_depos_loop) {
                    // ordering of axes form fastest to slowest:
                    // x/depos_order_1 to match deposition
                    // x%depos_order_1
                    // y
                    // z (not used)
                    // ppc
                    cellid = ((i_part * nz + uiz) * ny + uiy) * nx +
                    uix/depos_order_1 + ((uix%depos_order_1)*nx+depos_order_1-1)/depos_order_1;
                } else {
                    // ordering of axes form fastest to slowest:
                    // x
                    // y
                    // z (not used)
                    // ppc
                    cellid = ((i_part * nz + uiz) * ny + uiy) * nx + uix;
                }

                pcount[cellid] = 1;
            }
        });

        int num_to_add = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        m_init_num_par[mfi.tileIndex()] = new_size;

        if (num_to_add == 0) continue;

        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        auto arrdata = particle_tile.GetStructOfArrays().realarray();
        auto int_arrdata = particle_tile.GetStructOfArrays().intarray();

        int procID = amrex::ParallelDescriptor::MyProc();
        int pid = ParticleType::NextID();
        ParticleType::NextID(pid + num_to_add);

        const int init_ion_lev = m_init_ion_lev;

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

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                unsigned int cellid = 0;
                if (outer_depos_loop) {
                    cellid = ((i_part * nz + uiz) * ny + uiy) * nx +
                    uix/depos_order_1 + ((uix%depos_order_1)*nx+depos_order_1-1)/depos_order_1;
                } else {
                    cellid = ((i_part * nz + uiz) * ny + uiy) * nx + uix;
                }

                const int pidx = int(poffset[cellid] - poffset[0]);

                amrex::Real r[3] = {0.,0.,0.};

                ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                const amrex::Real rsq = x*x + y*y;
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    rsq > a_radius*a_radius ||
                    rsq < a_hollow_core_radius*a_hollow_core_radius ||
                    density_func(x, y, c_t) < min_density) continue;

                amrex::Real u[3] = {0.,0.,0.};
                ParticleUtil::get_gaussian_random_momentum(u, a_u_mean, a_u_std, engine);

                ParticleType& p = pstruct[pidx];
                p.id()   = pid + pidx;
                p.cpu()  = procID;
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;

                arrdata[PlasmaIdx::w        ][pidx] = scale_fac * density_func(x, y, c_t);
                arrdata[PlasmaIdx::w0       ][pidx] = scale_fac;
                arrdata[PlasmaIdx::ux       ][pidx] = u[0] * c_light;
                arrdata[PlasmaIdx::uy       ][pidx] = u[1] * c_light;
                arrdata[PlasmaIdx::psi][pidx] = std::sqrt(1._rt+u[0]*u[0]+u[1]*u[1]+u[2]*u[2])-u[2];
                arrdata[PlasmaIdx::x_prev   ][pidx] = x;
                arrdata[PlasmaIdx::y_prev   ][pidx] = y;
#ifndef HIPACE_USE_AB5_PUSH
                arrdata[PlasmaIdx::ux_half_step][pidx] = u[0] * c_light;
                arrdata[PlasmaIdx::uy_half_step][pidx] = u[1] * c_light;
                arrdata[PlasmaIdx::psi_inv_half_step][pidx] = 1._rt/arrdata[PlasmaIdx::psi][pidx];
#else
                arrdata[PlasmaIdx::ux_prev ][pidx] = u[0] * c_light;
                arrdata[PlasmaIdx::uy_prev ][pidx] = u[1] * c_light;
                arrdata[PlasmaIdx::psi_prev][pidx] = arrdata[PlasmaIdx::psi][pidx];

#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int iforce = PlasmaIdx::Fx1; iforce <= PlasmaIdx::Fpsi5; ++iforce) {
                    arrdata[iforce][pidx] = 0._rt;
                }
#endif
                arrdata[PlasmaIdx::x0       ][pidx] = x;
                arrdata[PlasmaIdx::y0       ][pidx] = y;
                int_arrdata[PlasmaIdx::ion_lev][pidx] = init_ion_lev;
            }
        });
    }

    AMREX_ASSERT(OK());
}

void
PlasmaParticleContainer::
InitIonizationModule (const amrex::Geometry& geom,
                      PlasmaParticleContainer* product_pc)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitIonizationModule()");

    using namespace amrex::literals;

    if (!m_can_ionize) return;
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
    // partial dx calculation for QSA
    auto dx = geom.CellSizeArray();
    const amrex::Real dt = dx[2] / phys_const.c;

    m_adk_power.resize(ion_atomic_number);
    m_adk_prefactor.resize(ion_atomic_number);
    m_adk_exp_prefactor.resize(ion_atomic_number);

    amrex::Real* AMREX_RESTRICT ionization_energies = h_ionization_energies.data();
    amrex::Real* AMREX_RESTRICT p_adk_power = m_adk_power.data();
    amrex::Real* AMREX_RESTRICT p_adk_prefactor = m_adk_prefactor.data();
    amrex::Real* AMREX_RESTRICT p_adk_exp_prefactor = m_adk_exp_prefactor.data();

    for (int i=0; i<ion_atomic_number; ++i)
    {
        const amrex::Real n_eff = (i+1) * std::sqrt(UH/ionization_energies[i]);
        const amrex::Real C2 = std::pow(2,2*n_eff)/(n_eff*std::tgamma(n_eff+l_eff+1)
                         * std::tgamma(n_eff-l_eff));
        p_adk_power[i] = -(2*n_eff - 1);
        const amrex::Real Uion = ionization_energies[i];
        p_adk_prefactor[i] = dt * wa * C2 * ( Uion/(2*UH) )
            * std::pow(2*std::pow((Uion/UH),3./2)*Ea,2*n_eff - 1);
        p_adk_exp_prefactor[i] = -2./3 * std::pow( Uion/UH,3./2) * Ea;
    }
}
