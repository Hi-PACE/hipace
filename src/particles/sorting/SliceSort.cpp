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
    HIPACE_PROFILE("shiftSlippedParticles()");

    if (beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This) == 0) {
        // nothing to do
        return;
    }

    const auto ptd = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();

    // min_z is the lower end of WhichBeamSlice::This
    const amrex::Real min_z = geom.ProbLo(2) + (slice-geom.Domain().smallEnd(2))*geom.CellSize(2);

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<int, int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    const int num_particles = beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This);

    // count the number of invalid and slipped particles
    reduce_op.eval(
        num_particles, reduce_data,
        [=] AMREX_GPU_DEVICE (const int ip) -> ReduceTuple
        {
            if (ptd.id(ip) < 0) {
                return {1, 0};
            } else if (ptd.pos(2, ip) < min_z) {
                return {0, 1};
            } else {
                return {0, 0};
            }
        });

    ReduceTuple t = reduce_data.value();

    const int num_invalid = amrex::get<0>(t);
    const int num_slipped = amrex::get<1>(t);
    const int num_stay = beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This)
                         - num_invalid - num_slipped;

    if (num_invalid == 0 && num_slipped == 0) {
        // nothing to do
        return;
    }

    const int next_size = beam.getNumParticles(WhichBeamSlice::Next);

    // there shouldn't be any slipped particles already on WhichBeamSlice::Next
    AMREX_ALWAYS_ASSERT(beam.getNumParticlesIncludingSlipped(WhichBeamSlice::Next) == next_size);

    beam.resize(WhichBeamSlice::Next, next_size, num_slipped);

    BeamTile tmp{};
    tmp.resize(num_stay);

    const auto ptd_tmp = tmp.getParticleTileData();

    // copy valid non slipped particles to the tmp tile
    const int num_stay2 = amrex::Scan::PrefixSum<int> (num_particles,
        [=] AMREX_GPU_DEVICE (const int ip) -> int
        {
            return ptd.id(ip) >= 0 && ptd.pos(2, ip) >= min_z;
        },
        [=] AMREX_GPU_DEVICE (const int ip, const int s)
        {
            if (ptd.id(ip) >= 0 && ptd.pos(2, ip) >= min_z) {
                for (int j=0; j<ptd_tmp.NAR; ++j) {
                    ptd_tmp.rdata(j)[s] = ptd.rdata(j)[ip];
                }
                for (int j=0; j<ptd_tmp.NAI; ++j) {
                    ptd_tmp.idata(j)[s] = ptd.idata(j)[ip];
                }
            }
        },
        amrex::Scan::Type::exclusive);

    AMREX_ALWAYS_ASSERT(num_stay == num_stay2);

    const auto ptd_next = beam.getBeamSlice(WhichBeamSlice::Next).getParticleTileData();

    // copy valid slipped particles to WhichBeamSlice::Next
    const int num_slipped2 = amrex::Scan::PrefixSum<int> (num_particles,
        [=] AMREX_GPU_DEVICE (const int ip) -> int
        {
            return ptd.id(ip) >= 0 && ptd.pos(2, ip) < min_z;
        },
        [=] AMREX_GPU_DEVICE (const int ip, const int s)
        {
            if (ptd.id(ip) >= 0 && ptd.pos(2, ip) < min_z) {
                for (int j=0; j<ptd_next.NAR; ++j) {
                    ptd_next.rdata(j)[s+next_size] = ptd.rdata(j)[ip];
                }
                for (int j=0; j<ptd_next.NAI; ++j) {
                    ptd_next.idata(j)[s+next_size] = ptd.idata(j)[ip];
                }
            }
        },
        amrex::Scan::Type::exclusive);

    AMREX_ALWAYS_ASSERT(num_slipped == num_slipped2);

    beam.getBeamSlice(WhichBeamSlice::This).swap(tmp);
    beam.resize(WhichBeamSlice::This, num_stay, 0);

    // stream sync before tmp is deallocated
    amrex::Gpu::streamSynchronize();
}
