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
#include "particles/beam/BeamParticleContainer.H"
#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"

void
HistogramDepositionPlasma (PlasmaParticleContainer& plasma, DiagnosticData& fd, amrex::Real zmid)
{
    if (!(fd.m_base_diag_type == DiagnosticData::diag_type::histogram)) {
        return;
    }
    HIPACE_PROFILE("HistogramDepositionPlasma()");
    using namespace amrex::literals;

    auto hist1 = fd.m_hist_exe_q1;
    auto hist2 = fd.m_hist_exe_q2;
    auto histw = fd.m_hist_exe_w;

    const amrex::Real h1_pos_offset = GetPosOffset(0, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real h2_pos_offset = GetPosOffset(1, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real d1_inv = fd.m_geom_io.InvCellSize(0);
    const amrex::Real d2_inv = fd.m_geom_io.InvCellSize(1);

    const amrex::Box bin_box = fd.m_geom_io.Domain();
    const CheckDomainBounds realspace_bounds{fd.m_realspace_geom};

    const bool use_second_dim = fd.m_hist_num_dims == 2;

    const bool can_ionize = plasma.m_can_ionize;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        Array2<amrex::Real> arr = fd.m_hist_gpu_fab.array();

        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int ip) {
                if (!ptd.id(ip).is_valid()) {
                    return;
                }

                const amrex::Real xp = ptd.pos(0, ip);
                const amrex::Real yp = ptd.pos(1, ip);

                if (!realspace_bounds.contains(xp, yp)) {
                    return;
                }

                const amrex::Real uxp = ptd.rdata(PlasmaIdx::ux)[ip];
                const amrex::Real uyp = ptd.rdata(PlasmaIdx::uy)[ip];
                const amrex::Real psi = ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real wp = ptd.rdata(PlasmaIdx::w)[ip];
                const amrex::Real ion_level =
                    amrex::Real(can_ionize ? ptd.idata(PlasmaIdx::ion_lev)[ip] : 0);

                const amrex::Real psi_inv = 1._rt / psi;
                const amrex::Real gamma = plasma_gamma(uxp, uyp, psi, psi_inv, 0._rt);
                const amrex::Real uzp = plasma_uz(gamma, psi);

                const amrex::Real hw = gamma * psi_inv *
                    histw(xp, yp, zmid, uxp, uyp, uzp, gamma * psi_inv, wp, ion_level);
                if (hw == 0._rt) {
                    return;
                }
                const amrex::Real h1 =
                    hist1(xp, yp, zmid, uxp, uyp, uzp, gamma * psi_inv, wp, ion_level);
                amrex::Real h2 = 0.5_rt;
                if (use_second_dim) {
                    h2 = hist2(xp, yp, zmid, uxp, uyp, uzp, gamma * psi_inv, wp, ion_level);
                }

                const amrex::Real h1_mid = (h1 - h1_pos_offset) * d1_inv;
                const amrex::Real h2_mid = (h2 - h2_pos_offset) * d2_inv;

                const int i = static_cast<int>(std::floor(h1_mid + 0.5_rt));
                const int j = static_cast<int>(std::floor(h2_mid + 0.5_rt));

                if (bin_box.smallEnd(0) <= i && i <= bin_box.bigEnd(0) &&
                    bin_box.smallEnd(1) <= j && j <= bin_box.bigEnd(1)) {
                    amrex::Gpu::Atomic::Add(arr.ptr(i, j), hw);
                }
            }
        );
    }
}

void
HistogramDepositionBeam (BeamParticleContainer& beam, DiagnosticData& fd)
{
    if (!(fd.m_base_diag_type == DiagnosticData::diag_type::histogram)) {
        return;
    }
    HIPACE_PROFILE("HistogramDepositionBeam()");
    using namespace amrex::literals;

    auto hist1 = fd.m_hist_exe_q1;
    auto hist2 = fd.m_hist_exe_q2;
    auto histw = fd.m_hist_exe_w;

    const amrex::Real h1_pos_offset = GetPosOffset(0, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real h2_pos_offset = GetPosOffset(1, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real d1_inv = fd.m_geom_io.InvCellSize(0);
    const amrex::Real d2_inv = fd.m_geom_io.InvCellSize(1);

    const amrex::Box bin_box = fd.m_geom_io.Domain();
    const CheckDomainBounds realspace_bounds{fd.m_realspace_geom};

    const bool use_second_dim = fd.m_hist_num_dims == 2;

    // Loop over particle boxes
    auto ptd = beam.getBeamSlice(WhichBeamSlice::This).getConstParticleTileData();
    Array2<amrex::Real> arr = fd.m_hist_gpu_fab.array();

    amrex::ParallelFor(beam.getNumParticles(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (int ip) {
            if (!ptd.id(ip).is_valid()) {
                return;
            }

            const amrex::Real xp = ptd.pos(0, ip);
            const amrex::Real yp = ptd.pos(1, ip);
            const amrex::Real zp = ptd.pos(2, ip);

            if (!realspace_bounds.contains(xp, yp)) {
                return;
            }

            const amrex::Real uxp = ptd.rdata(BeamIdx::ux)[ip];
            const amrex::Real uyp = ptd.rdata(BeamIdx::uy)[ip];
            const amrex::Real uzp = ptd.rdata(BeamIdx::uz)[ip];
            const amrex::Real wp = ptd.rdata(BeamIdx::w)[ip];
            const amrex::Real gamma = amrex::Math::rsqrt(1._rt + uxp*uxp + uyp*uyp + uzp*uzp);
            const amrex::Real gamma_psi = gamma / (gamma - uzp);
            const int ion_level = 0;

            const amrex::Real hw = histw(xp, yp, zp, uxp, uyp, uzp, gamma_psi, wp, ion_level);
            if (hw == 0._rt) {
                return;
            }
            const amrex::Real h1 = hist1(xp, yp, zp, uxp, uyp, uzp, gamma_psi, wp, ion_level);
            amrex::Real h2 = 0.5_rt;
            if (use_second_dim) {
                h2 = hist2(xp, yp, zp, uxp, uyp, uzp, gamma_psi, wp, ion_level);
            }

            const amrex::Real h1_mid = (h1 - h1_pos_offset) * d1_inv;
            const amrex::Real h2_mid = (h2 - h2_pos_offset) * d2_inv;

            const int i = static_cast<int>(std::floor(h1_mid + 0.5_rt));
            const int j = static_cast<int>(std::floor(h2_mid + 0.5_rt));

            if (bin_box.smallEnd(0) <= i && i <= bin_box.bigEnd(0) &&
                bin_box.smallEnd(1) <= j && j <= bin_box.bigEnd(1)) {
                amrex::Gpu::Atomic::Add(arr.ptr(i, j), hw);
            }
        }
    );
}
