/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#include "TileSort.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_ParticleTransformation.H>

PlasmaBins
findParticlesInEachTile (
    amrex::Box bx, int bin_size,
    PlasmaParticleContainer& plasma, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("findParticlesInEachTile()");

    // Tile box: only 1 cell longitudinally, same as bx transversally.
    const amrex::Box tcbx = bx.coarsen(bin_size);
    const amrex::Box cbx = {{tcbx.smallEnd(0),tcbx.smallEnd(1),0}, {tcbx.bigEnd(0),tcbx.bigEnd(1),0}};

    PlasmaBins bins;

    // Extract particle structures for this tile
    int count = 0; // number of boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {
        count += 1;

        // Extract box properties
        const auto lo = lbound(cbx);
        const auto dxi = amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>({
                geom.InvCellSizeArray()[0]/bin_size,
                geom.InvCellSizeArray()[1]/bin_size,
                1.});
        const auto plo = geom.ProbLoArray();

        // Find the particles that are in each tile and return collections of indices per tile.
        bins.build(
            pti.numParticles(), pti.GetParticleTile().getParticleTileData(), cbx,
            // Pass lambda function that returns the tile index
            [=] AMREX_GPU_DEVICE (const PlasmaParticleContainer::ParticleType& p)
            noexcept -> amrex::IntVect
            {
                return amrex::IntVect(
                    AMREX_D_DECL(
                        static_cast<int>((p.pos(0)-plo[0])*dxi[0]-lo.x),
                        static_cast<int>((p.pos(1)-plo[1])*dxi[1]-lo.y),
                        0));
            });
    }
    AMREX_ALWAYS_ASSERT(count <= 1);
    return bins;
}

BeamBins
findBeamParticlesInEachTile (
    amrex::Box bx, int bin_size, int islice_local,
    BeamParticleContainer& beam, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("findBeamParticlesInEachTile()");

    // Tile box: only 1 cell longitudinally, same as bx transversally.
    const amrex::Box tcbx = bx.coarsen(bin_size);
    const amrex::Box cbx = {{tcbx.smallEnd(0),tcbx.smallEnd(1),0}, {tcbx.bigEnd(0),tcbx.bigEnd(1),0}};

    BeamBins bins;

    const int box_offset = beam.m_box_sorter.boxOffsetsPtr()[beam.m_ibox];

    BeamBins::index_type const * const indices = beam.m_slice_bins.permutationPtr();
    BeamBins::index_type const * const slice_offsets = beam.m_slice_bins.offsetsPtrCpu();
    BeamBins::index_type const
        cell_start = slice_offsets[islice_local], cell_stop = slice_offsets[islice_local+1];
    // The particles that are in slice islice_local are
    // given by the indices[cell_start:cell_stop]

    int const num_particles = cell_stop-cell_start;

    // Extract box properties
    const auto lo = lbound(cbx);
    const auto dxi = amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>({
            geom.InvCellSizeArray()[0]/bin_size,
            geom.InvCellSizeArray()[1]/bin_size,
            1.});
    const auto plo = geom.ProbLoArray();

    // Find the particles that are in each slice and return collections of indices per slice.
    bins.build(
        num_particles, beam.getParticleTileData(), cbx,
        // Pass lambda function that returns the slice index
        [=] AMREX_GPU_DEVICE (const BeamParticleContainer::ParticleTileDataType& pdt,
                              unsigned int idx)
        noexcept -> amrex::IntVect
        {
            auto p = pdt[indices[cell_start+idx] + box_offset];

            return amrex::IntVect(
                AMREX_D_DECL(
                    static_cast<int>((p.pos(0)-plo[0])*dxi[0]-lo.x),
                    static_cast<int>((p.pos(1)-plo[1])*dxi[1]-lo.y),
                    0));
        });

    return bins;
}
