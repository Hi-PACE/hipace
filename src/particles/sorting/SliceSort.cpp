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

void
shiftSlippedParticles (BeamParticleContainer& beam, const int slice, amrex::Geometry const& geom)
{
    if (beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This) == 0) {
        // nothing to do
        return;
    }

    HIPACE_PROFILE("shiftSlippedParticles()");

    // remove all invalid particles from WhichBeamSlice::This (including slipped)
    amrex::removeInvalidParticles(beam.getBeamSlice(WhichBeamSlice::This));

    // min_z is the lower end of WhichBeamSlice::This
    const amrex::Real min_z = geom.ProbLo(2) + (slice-geom.Domain().smallEnd(2))*geom.CellSize(2);

    // put non slipped particles at the start of the slice
    const int num_stay = amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (auto& ptd, int i) {
            return ptd.pos(2, i) >= min_z;
        });

    const int num_slipped = beam.getBeamSlice(WhichBeamSlice::This).size() - num_stay;

    if (num_slipped == 0) {
        // nothing to do
        beam.resize(WhichBeamSlice::This, num_stay, 0);
        return;
    }

    const int next_size = beam.getNumParticles(WhichBeamSlice::Next);

    // there shouldn't be any slipped particles already on WhichBeamSlice::Next
    AMREX_ALWAYS_ASSERT(beam.getNumParticlesIncludingSlipped(WhichBeamSlice::Next) == next_size);

    beam.resize(WhichBeamSlice::Next, next_size, num_slipped);

    const auto ptd_this = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();
    const auto ptd_next = beam.getBeamSlice(WhichBeamSlice::Next).getParticleTileData();

    amrex::ParallelFor(num_slipped,
        [=] AMREX_GPU_DEVICE (int i)
        {
            // copy particles from WhichBeamSlice::This to WhichBeamSlice::Next
            amrex::copyParticle(ptd_next, ptd_this, num_stay + i, next_size + i);
        });


    // stream sync before WhichBeamSlice::This is resized
    amrex::Gpu::streamSynchronize();

    beam.resize(WhichBeamSlice::This, num_stay, 0);
}
