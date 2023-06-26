/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#include "BoxSort.H"
#include "particles/beam/BeamParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_ParticleTransformation.H>

void BoxSorter::sortParticlesByBox (BeamParticleContainer& a_beam,
                                    const amrex::Box bx, const amrex::Geometry& a_geom)
{
    HIPACE_PROFILE("sortBeamParticlesByBox()");

    int const np = a_beam.getBeamInitSlice().numParticles();
    auto ptd = a_beam.getBeamInitSlice().getParticleTileData();

    int num_boxes = bx.length(2);
    m_box_counts.resize(num_boxes+1);
    m_box_counts.assign(num_boxes+1, 0);
    m_box_counts_cpu.resize(num_boxes+1);
    m_box_offsets.resize(num_boxes+1);
    m_box_offsets.assign(num_boxes+1, 0);
    m_box_offsets_cpu.resize(num_boxes+1);

    amrex::Gpu::DeviceVector<unsigned int> dst_indices(np);

    auto p_box_counts = m_box_counts.dataPtr();
    auto p_dst_indices = dst_indices.dataPtr();

    // Extract box properties
    const int lo_z = bx.smallEnd(2);
    const amrex::Real dzi = a_geom.InvCellSize(2);
    const amrex::Real plo_z = a_geom.ProbLo(2);

    AMREX_FOR_1D ( np, i,
    {
        int dst_box = static_cast<int>((ptd.pos(2, i) - plo_z) * dzi - lo_z);
        if (ptd.id(i) < 0) dst_box = num_boxes; // if pid is invalid, remove particle
        if (dst_box < 0) {
            // particle has left domain transversely, stick it at the end and invalidate
            dst_box = num_boxes;
            ptd.id(i) = -std::abs(ptd.id(i));
        }
        unsigned int index = amrex::Gpu::Atomic::Add(&p_box_counts[dst_box], 1u);
        p_dst_indices[i] = index;
    });

    amrex::Gpu::exclusive_scan(m_box_counts.begin(), m_box_counts.end(), m_box_offsets.begin());

    BeamTileInit tmp{};
    tmp.resize(np);

    auto p_box_offsets = m_box_offsets.dataPtr();
    AMREX_FOR_1D ( np, i,
    {
        int dst_box = static_cast<int>((ptd.pos(2, i) - plo_z) * dzi - lo_z);
        if (ptd.id(i) < 0) dst_box = num_boxes; // if pid is invalid, remove particle
        if (dst_box < 0) dst_box = num_boxes;
        p_dst_indices[i] += p_box_offsets[dst_box];
    });

    amrex::scatterParticles<BeamTileInit>(tmp, a_beam.getBeamInitSlice(), np, dst_indices.dataPtr());

    a_beam.getBeamInitSlice().swap(tmp);
#ifdef AMREX_USE_GPU
    amrex::Gpu::dtoh_memcpy_async(m_box_counts_cpu.dataPtr(), m_box_counts.dataPtr(),
                                  m_box_counts.size() * sizeof(index_type));

    amrex::Gpu::dtoh_memcpy_async(m_box_offsets_cpu.dataPtr(), m_box_offsets.dataPtr(),
                                  m_box_offsets.size() * sizeof(index_type));

    amrex::Gpu::streamSynchronize();
#else
    std::memcpy(m_box_counts_cpu.dataPtr(), m_box_counts.dataPtr(),
                m_box_counts.size() * sizeof(index_type));

    std::memcpy(m_box_offsets_cpu.dataPtr(), m_box_offsets.dataPtr(),
                m_box_offsets.size() * sizeof(index_type));
#endif
}
