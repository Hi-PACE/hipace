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
                    const int which_slice,
                    amrex::Vector<amrex::Geometry> const& gm, 
                    int const lev)
{
    if (!Hipace::m_deposit_temp || !Hipace::m_deposit_temp_individual) { // deposit temperature in input
        return;
    }
    HIPACE_PROFILE("TemperatureDeposition_PlasmaParticleContainer()");
    using namespace amrex::literals;

    // only deposit ux individual on WhichSlice::This
    const bool deposit_temp_individual = true;
    const std::string ux_str = deposit_temp_individual ? "ux_" + plasma.GetName() : "ux";

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
        const amrex::Real dz_inv = gm[lev].InvCellSize(2);
        // in normalized units this is rescaling dx and dy for MR,
        // while in SI units it's the factor for charge to charge density
        const amrex::Real invvol = Hipace::m_normalized_units ?
            gm[0].CellSize(0)*gm[0].CellSize(1)*dx_inv*dy_inv
            : dx_inv*dy_inv*dz_inv;

        const PhysConst pc = get_phys_const();
        const amrex::Real clight = pc.c;
        const amrex::Real clightinv = 1.0_rt/pc.c;
        const amrex::Real clightinv2 = clightinv*clightinv;
        const bool can_ionize = plasma.m_can_ionize;
        const bool use_laser = Hipace::m_use_laser;
        const amrex::Real laser_norm = (plasma.m_charge/pc.q_e) * (pc.m_e/plasma.m_mass)
            * (plasma.m_charge/pc.q_e) * (pc.m_e/plasma.m_mass);

        // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab

        SharedMemoryDeposition<1, 1, true>(
            int(pti.numParticles()),
            // is_valid
            // return whether the particle is valid and should deposit
            [=] AMREX_GPU_DEVICE (int ip, auto ptd)
            {
            // only deposit plasma currents on or below their according MR level
                return ptd.id(ip).is_valid() && (lev == 0 || ptd.cpu(ip) >= lev);
            },
            // get_cell
            // return the lowest cell index that the particle deposits into
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

                return {i, j};
            },
            // deposit
            [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                                  Array3<amrex::Real> arr,
                                  auto cache_idx, auto depos_idx) noexcept
            {   
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);

                amrex::Real Aabssqp = 0._rt;
                amrex::Real laser_norm_ion = laser_norm;
                if (can_ionize) {
                    laser_norm_ion *=
                        ptd.idata(PlasmaIdx::ion_lev)[ip] * ptd.idata(PlasmaIdx::ion_lev)[ip];
                }
                if (use_laser) {
                    doLaserGatherShapeN<0>(xp, yp, Aabssqp, arr, cache_idx[0],
                                                    dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                    Aabssqp *= laser_norm_ion;
                }

                const amrex::Real ux = ptd.rdata(PlasmaIdx::ux)[ip];
                const amrex::Real uy = ptd.rdata(PlasmaIdx::uy)[ip];
                amrex::Real psi = ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real uz = (1._rt + ux*ux*clightinv2 + uy*uy*clightinv2 
                    + 0.5_rt*Aabssqp - psi*psi)/(2.*psi) * clight;
                const amrex::Real w = ptd.rdata(PlasmaIdx::w)[ip];

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                // --- Compute shape factors
                // x direction
                auto [shape_x, i] =
                compute_single_shape_factor<false, 0>(xmid, 0);

                // y direction
                auto [shape_y, j] =
                compute_single_shape_factor<false, 0>(ymid, 0);

                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[0]), w);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[1]), w*ux);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[2]), w*uy);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[3]), w*uz);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[4]), w*ux*ux);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[5]), w*uy*uy);
                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[6]), w*uz*uz);
            },
            isl_fab.array(),
            isl_fab.box(), pti.GetParticleTile().getParticleTileData(),
            amrex::GpuArray<int, 0>{},
            amrex::GpuArray<int, 7>{w, ux, uy, uz, uxsq, uysq, uzsq});
    }
}