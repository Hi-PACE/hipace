/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "SliceSort.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

#include <AMReX_ParticleTransformation.H>

void
findParticlesInEachSlice (
    [[maybe_unused]] int ibox, [[maybe_unused]] amrex::Box bx, [[maybe_unused]] BeamParticleContainer& beam, [[maybe_unused]] amrex::Geometry const& geom)
{
    HIPACE_PROFILE("findParticlesInEachSlice()");

    // Slice box: only 1 cell transversally, same as bx longitudinally.
    amrex::Box cbx ({0,0,bx.smallEnd(2)}, {0,0,bx.bigEnd(2)});

    const int np = beam.m_box_sorter.boxCountsPtr()[ibox];
    const int offset = beam.m_box_sorter.boxOffsetsPtr()[ibox];

    AMREX_ALWAYS_ASSERT(offset==0);

    // Extract box properties
    const auto lo = lbound(cbx);
    const auto dxi = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();

    // Find the particles that are in each slice and return collections of indices per slice.
    beam.m_slice_bins.build(
        np, beam.getParticleTileData(), cbx,
        // Pass lambda function that returns the slice index
        [=] AMREX_GPU_DEVICE (const BeamParticleContainer::ParticleType& p)
        noexcept -> amrex::IntVect
        {
            return amrex::IntVect(
                AMREX_D_DECL(0, 0, static_cast<int>((p.pos(2)-plo[2])*dxi[2]-lo.z)));
        });
}
