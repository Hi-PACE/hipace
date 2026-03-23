/* Copyright 2026
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "HistogramDeposition.H"
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
HistogramDeposition (PlasmaParticleContainer& plasma,
                     DiagnosticData& fd)
{
    if (!(fd.m_base_diag_type == DiagnosticData::diag_type::histogram)) {
        return;
    }
    HIPACE_PROFILE("TemperatureDeposition_PlasmaParticleContainer()");
    using namespace amrex::literals;

    auto hist1 = [=] AMREX_GPU_DEVICE () {
        return 1._rt;
    };

    auto hist2 = [=] AMREX_GPU_DEVICE () {
        return 2._rt;
    };

    const amrex::Real h1_pos_offset = 0._rt;
    const amrex::Real h2_pos_offset = 0._rt;
    const amrex::Real d1_inv = 0._rt;
    const amrex::Real d2_inv = 0._rt;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {

        // Loop over particles
        SharedMemoryDeposition<1, 1, true>(
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int ip, auto ptd)
            {
                return ptd.id(ip).is_valid();
            },
            [=] AMREX_GPU_DEVICE (int ip, auto ptd) -> amrex::IntVectND<2>
            {
                const amrex::Real h1mid = (hist1() - h1_pos_offset) * d1_inv;
                const amrex::Real h2mid = (hist2() - h2_pos_offset) * d2_inv;

                auto [shape_h1, i] = shape_factor<0>(h1mid, 0);
                auto [shape_h2, j] = shape_factor<0>(h2mid, 0);

                return {i, j};
            },
            [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                                  Array3<amrex::Real> arr,
                                  auto cache_idx, auto depos_idx) noexcept
            {
                const amrex::Real uxp = ptd.rdata(PlasmaIdx::ux)[ip];
                const amrex::Real uyp = ptd.rdata(PlasmaIdx::uy)[ip];
                const amrex::Real psi_inv = 1._rt / ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real gamma_psi = plasma_gamma_psi(uxp, uyp, psi_inv, 0._rt);
                const amrex::Real wp = ptd.rdata(PlasmaIdx::w)[ip] * gamma_psi;

                const amrex::Real h1mid = (hist1() - h1_pos_offset) * d1_inv;
                const amrex::Real h2mid = (hist2() - h2_pos_offset) * d2_inv;

                // --- Compute shape factors
                auto [shape_h1, i] = shape_factor<0>(h1mid, 0);
                auto [shape_h2, j] = shape_factor<0>(h2mid, 0);

                amrex::Gpu::Atomic::Add(arr.ptr(i, j, depos_idx[0]), shape_h1*shape_h2*wp);
            },
            fd.m_hist_gpu_fab.array(),
            fd.m_hist_gpu_fab.box(), pti.GetParticleTile().getParticleTileData(),
            amrex::GpuArray<int, 0>{},
            amrex::GpuArray<int, 1>{0}
        );
    }
}
