/* Copyright 2021-2022 Axel Huebl, MaxThevenet, Severin Diederichs
 *
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SliceSort.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

#include <AMReX_ParticleTransformation.H>

BeamBins
findParticlesInEachSlice (
    int lev, int ibox, amrex::Box bx, BeamParticleContainer& beam,
    amrex::Vector<amrex::Geometry> const& geom, const BoxSorter& a_box_sorter)
{
    HIPACE_PROFILE("findParticlesInEachSlice()");

    // Slice box: only 1 cell transversally, same as bx longitudinally.
    amrex::Box cbx ({0,0,bx.smallEnd(2)}, {0,0,bx.bigEnd(2)});
    if (lev == 1) cbx.refine(Hipace::GetRefRatio(lev));

    const int np = a_box_sorter.boxCountsPtr()[ibox];
    const int offset = a_box_sorter.boxOffsetsPtr()[ibox];

    // Extract particle structures for this tile
    BeamParticleContainer::ParticleType const* particle_ptr = beam.GetArrayOfStructs()().data();
    particle_ptr += offset;

    // Extract box properties
    const auto lo = lbound(cbx);
    const auto dxi = geom[lev].InvCellSizeArray();
    const auto plo = geom[lev].ProbLoArray();

    // Find the particles that are in each slice and return collections of indices per slice.
    BeamBins bins;
    bins.build(
        np, particle_ptr, cbx,
        // Pass lambda function that returns the slice index
        [=] AMREX_GPU_DEVICE (const BeamParticleContainer::ParticleType& p)
        noexcept -> amrex::IntVect
        {
            return amrex::IntVect(
                AMREX_D_DECL(0, 0, static_cast<int>((p.pos(2)-plo[2])*dxi[2]-lo.z)));
        });

    return bins;
}
