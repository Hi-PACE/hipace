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

void BoxSorter::sortParticlesByBox (BeamParticleContainer& a_beam, const amrex::Geometry& a_geom)
{
    HIPACE_PROFILE("sortBeamParticlesByBox()");

    m_box_permutations.setArena(
        a_beam.m_initialize_on_cpu ? amrex::The_Pinned_Arena() : amrex::The_Arena());

    const index_type np = a_beam.getBeamInitSlice().numParticles();
    auto ptd = a_beam.getBeamInitSlice().getParticleTileData();

    int num_boxes = a_geom.Domain().length(2);
    m_box_counts_cpu.resize(num_boxes+1);
    m_box_offsets_cpu.resize(num_boxes+1);
    m_box_permutations.resize(np);

    amrex::PODVector<index_type, amrex::PolymorphicArenaAllocator<index_type>> local_offsets {};
    local_offsets.setArena(
        a_beam.m_initialize_on_cpu ? amrex::The_Pinned_Arena() : amrex::The_Arena());
    local_offsets.resize(np);

    amrex::Gpu::DeviceVector<index_type> box_counts (num_boxes+1, 0);
    amrex::Gpu::DeviceVector<index_type> box_offsets (num_boxes+1, 0);

    auto p_box_counts = box_counts.dataPtr();
    auto p_local_offsets = local_offsets.dataPtr();
    auto p_permutations = m_box_permutations.dataPtr();

    // Extract box properties
    const amrex::Real dzi = a_geom.InvCellSize(2);
    const amrex::Real plo_z = a_geom.ProbLo(2);

    amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (const index_type i) {
            int dst_box = static_cast<int>((ptd.pos(2, i) - plo_z) * dzi);
            if (ptd.id(i) < 0) dst_box = num_boxes; // if pid is invalid, remove particle
            if (dst_box < 0 || dst_box > num_boxes) {
                // particle has left domain transversely, stick it at the end and invalidate
                dst_box = num_boxes;
                ptd.id(i) = -std::abs(ptd.id(i));
            }
            unsigned int index = amrex::Gpu::Atomic::Add(&p_box_counts[dst_box], 1u);
            p_local_offsets[i] = index;
        });

    amrex::Gpu::exclusive_scan(box_counts.begin(), box_counts.end(), box_offsets.begin());

    auto p_box_offsets = box_offsets.dataPtr();

    amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (const index_type i) {
            int dst_box = static_cast<int>((ptd.pos(2, i) - plo_z) * dzi);
            if (ptd.id(i) < 0) dst_box = num_boxes; // if pid is invalid, remove particle
            if (dst_box < 0 || dst_box > num_boxes) dst_box = num_boxes;
            p_permutations[p_local_offsets[i] + p_box_offsets[dst_box]] = i;
        });

#ifdef AMREX_USE_GPU
    amrex::Gpu::dtoh_memcpy_async(m_box_counts_cpu.dataPtr(), box_counts.dataPtr(),
                                  box_counts.size() * sizeof(index_type));

    amrex::Gpu::dtoh_memcpy_async(m_box_offsets_cpu.dataPtr(), box_offsets.dataPtr(),
                                  box_offsets.size() * sizeof(index_type));

    amrex::Gpu::streamSynchronize();
#else
    std::memcpy(m_box_counts_cpu.dataPtr(), box_counts.dataPtr(),
                box_counts.size() * sizeof(index_type));

    std::memcpy(m_box_offsets_cpu.dataPtr(), box_offsets.dataPtr(),
                box_offsets.size() * sizeof(index_type));
#endif
}
