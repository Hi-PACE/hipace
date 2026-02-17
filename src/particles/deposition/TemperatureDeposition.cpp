/* Copyright 2025
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, EyaDammak
 * License: BSD-3-Clause-LBNL
 */
#include "TemperatureDeposition.H"
#include "DepositionUtil.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "particles/particles_utils/FieldGather.H"
#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"

void
DepositTemperature (PlasmaParticleContainer& plasma,
                    Fields & fields,
                    amrex::Vector<amrex::Geometry> const& gm,
                    int const lev)
{
    if (!Hipace::m_deposit_temp_individual) { // deposit temperature in input
        return;
    }
    HIPACE_PROFILE("TemperatureDeposition_PlasmaParticleContainer()");
    using namespace amrex::literals;


    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {
        // Create the map with weight, momentum and squared momentum
        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];

        const int w = Comps[WhichSlice::This]["w_" + plasma.GetName()];
        const int ux = Comps[WhichSlice::This]["ux_" + plasma.GetName()];
        const int uy = Comps[WhichSlice::This]["uy_" + plasma.GetName()];
        const int uz = Comps[WhichSlice::This]["uz_" + plasma.GetName()];
        const int uxsq = Comps[WhichSlice::This]["ux^2_" + plasma.GetName()];
        const int uysq = Comps[WhichSlice::This]["uy^2_" + plasma.GetName()];
        const int uzsq = Comps[WhichSlice::This]["uz^2_" + plasma.GetName()];

        // Offset for converting positions to indexes
        const amrex::Real x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

        // Extract box properties
        const amrex::Real dx_inv = gm[lev].InvCellSize(0);
        const amrex::Real dy_inv = gm[lev].InvCellSize(1);
        // extract laser properties and boolean for the presence of the laser and for ionization
        const PhysConst pc = get_phys_const();
        const int aabs = Hipace::m_use_laser ? Comps[WhichSlice::This]["aabs"] : -1;
        const bool can_ionize = plasma.m_can_ionize;
        const bool use_laser = Hipace::m_use_laser;
        const amrex::Real laser_norm = (plasma.m_charge/pc.q_e) * (pc.m_e/plasma.m_mass)
            * (plasma.m_charge/pc.q_e) * (pc.m_e/plasma.m_mass);

        // Loop over particles
        SharedMemoryDeposition<3, 3, true>(
            int(pti.numParticles()),
            // is_valid
            // return whether the particle is valid and should deposit
            [=] AMREX_GPU_DEVICE (int ip, auto ptd)
            {
            // only deposit on or below their according MR level
                return ptd.id(ip).is_valid() && (lev == 0 || ptd.cpu(ip) >= lev);
            },
            [=] AMREX_GPU_DEVICE (int ip, auto ptd) -> amrex::IntVectND<2>
            {
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                auto [shape_x, i] =
                compute_single_shape_factor<false, 0>(xmid, 0);

                auto [shape_y, j] =
                compute_single_shape_factor<false, 0>(ymid, 0);

                return {i-1, j-1};
            },
            // deposit of weight, momentum (ux, uy, uz) and their squares (uxsq, uysq, uzsq)
            [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                                  Array3<amrex::Real> arr,
                                  auto cache_idx, auto depos_idx) noexcept
            {
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);

                amrex::Real Aabssqp = 0._rt;
                if (use_laser) {
                    amrex::Real laser_norm_ion = laser_norm;
                    if (can_ionize) {
                        laser_norm_ion *=
                            ptd.idata(PlasmaIdx::ion_lev)[ip] * ptd.idata(PlasmaIdx::ion_lev)[ip];
                    }
                    doLaserGatherShapeN<2>(xp, yp, Aabssqp, arr, cache_idx[0],
                                           dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                    Aabssqp *= laser_norm_ion;
                }

                const amrex::Real uxp = ptd.rdata(PlasmaIdx::ux)[ip];
                const amrex::Real uyp = ptd.rdata(PlasmaIdx::uy)[ip];
                const amrex::Real psi = ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real psi_inv = 1._rt / psi;
                const amrex::Real gamma = plasma_gamma(uxp, uyp, psi, psi_inv, Aabssqp);
                const amrex::Real uzp = plasma_uz(gamma, psi);
                const amrex::Real wp = ptd.rdata(PlasmaIdx::w)[ip] * gamma * psi_inv;

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                // --- Compute shape factors
                // x direction
                auto [shape_x, i] = compute_single_shape_factor<false, 0>(xmid, 0);
                // y direction
                auto [shape_y, j] = compute_single_shape_factor<false, 0>(ymid, 0);

                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[0]), wp);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[1]), wp*uxp);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[2]), wp*uyp);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[3]), wp*uzp);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[4]), wp*uxp*uxp);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[5]), wp*uyp*uyp);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[6]), wp*uzp*uzp);
            },
            isl_fab.array(),
            isl_fab.box(), pti.GetParticleTile().getParticleTileData(),
            amrex::GpuArray<int, 1>{aabs},
            amrex::GpuArray<int, 7>{w, ux, uy, uz, uxsq, uysq, uzsq}
        );
        Array3<amrex::Real> field_arr = isl_fab.array();

        // Normalize the components of momentum (ux, uy, uz) and their squares (uxsq, uysq, uzsq)
        // by dividing them by the total weight (w) in each cell. If the weight is zero, no division is performed.
        amrex::ParallelFor(
            to2D(isl_fab.box()),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
                {
                    amrex::Real wp_inv = field_arr(i, j, w) == amrex::Real{0} ? amrex::Real{0} : amrex::Real{1} / field_arr(i, j, w);
                    field_arr(i, j, ux) *= wp_inv;
                    field_arr(i, j, uy) *= wp_inv;
                    field_arr(i, j, uz) *= wp_inv;
                    field_arr(i, j, uxsq) *= wp_inv;
                    field_arr(i, j, uysq) *= wp_inv;
                    field_arr(i, j, uzsq) *= wp_inv;
                }
        );
    }
}
