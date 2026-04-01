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

    auto hist1 = fd.m_hist_exe_q1;
    auto hist2 = fd.m_hist_exe_q2;
    auto histw = fd.m_hist_exe_w;

    const amrex::Real h1_pos_offset = GetPosOffset(0, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real h2_pos_offset = GetPosOffset(0, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real d1_inv = fd.m_geom_io.InvCellSize(0);
    const amrex::Real d2_inv = fd.m_geom_io.InvCellSize(1);

    const bool can_ionize = plasma.m_can_ionize;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        Array2<amrex::Real> arr = fd.m_hist_gpu_fab.array();

        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int ip) {
                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);
                const amrex::Real uxp = ptd.rdata(PlasmaIdx::ux)[ip];
                const amrex::Real uyp = ptd.rdata(PlasmaIdx::uy)[ip];
                const amrex::Real psi = ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real wp = ptd.rdata(PlasmaIdx::w)[ip];
                const amrex::Real ion_level =
                    amrex::Real(can_ionize ? ptd.idata(PlasmaIdx::ion_lev)[ip] : 0);

                const amrex::Real psi_inv = 1._rt / psi;
                const amrex::Real gamma = plasma_gamma(uxp, uyp, psi, psi_inv, 0._rt);
                const amrex::Real uzp = plasma_uz(gamma, psi);

                const amrex::Real h1 = hist1(xp, yp, uxp, uyp, uzp, gamma * psi_inv, wp, ion_level);
                const amrex::Real h2 = hist2(xp, yp, uxp, uyp, uzp, gamma * psi_inv, wp, ion_level);
                const amrex::Real hw = histw(xp, yp, uxp, uyp, uzp, gamma * psi_inv, wp, ion_level);

                const amrex::Real h1_mid = (h1 - h1_pos_offset) * d1_inv;
                const amrex::Real h2_mid = (h2 - h2_pos_offset) * d2_inv;

                auto [shape_h1, i] = shape_factor<0>(h1_mid, 0);
                auto [shape_h2, j] = shape_factor<0>(h2_mid, 0);

                amrex::Gpu::Atomic::Add(arr.ptr(i, j), hw);
            }
        );
    }
}
