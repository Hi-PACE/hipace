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

    const int num_particles = beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This);
    const auto ptdr = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();
    // min_z is the lower end of WhichBeamSlice::This
    const amrex::Real min_z = geom.ProbLo(2) + (slice-geom.Domain().smallEnd(2))*geom.CellSize(2);

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<int, int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    // count the number of invalid and slipped particles
    reduce_op.eval(
        num_particles, reduce_data,
        [=] AMREX_GPU_DEVICE (const int i) -> ReduceTuple
        {
            if (!ptdr.id(i).is_valid()) {
                return {0, 0};
            } else if (ptdr.pos(2, i) >= min_z) {
                return {1, 0};
            } else {
                return {1, 1};
            }
        });

    const auto [num_valid, num_slipped] = reduce_data.value();
    const int num_stay = num_valid - num_slipped;

    if (num_valid != num_particles) {
        // remove all invalid particles from WhichBeamSlice::This (including slipped)
        amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This), num_valid,
            [=] AMREX_GPU_DEVICE (auto& ptd, int i) {
                return ptd.id(i).is_valid();
            });

        beam.getBeamSlice(WhichBeamSlice::This).resize(num_valid);
    }

    if (num_slipped == 0) {
        // nothing to do
        beam.resize(WhichBeamSlice::This, num_stay, 0);
        return;
    }

    // put non slipped particles at the start of the slice
    amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This), num_stay,
        [=] AMREX_GPU_DEVICE (auto& ptd, int i) {
            return ptd.pos(2, i) >= min_z;
        });

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
