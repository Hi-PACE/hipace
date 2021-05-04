#include "BinSort.H"

#include <AMReX_ParticleTransformation.H>

BeamBins
findParticlesInEachSlice (
    int /*lev*/, int ibox, amrex::Box bx,
    BeamParticleContainer& beam, const amrex::Geometry& geom,
    const BoxSorter& a_box_sorter)
{
    // Slice box: only 1 cell transversally, same as bx longitudinally.
    const amrex::Box cbx ({0,0,bx.smallEnd(2)}, {0,0,bx.bigEnd(2)});

    const int np = a_box_sorter.boxCountsPtr()[ibox];
    const int offset = a_box_sorter.boxOffsetsPtr()[ibox];

    // Extract particle structures for this tile
    BeamParticleContainer::ParticleType const* particle_ptr = beam.GetArrayOfStructs()().data();
    particle_ptr += offset;

    // Extract box properties
    const auto lo = lbound(cbx);
    const auto dxi = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();

    // Find the particles that are in each slice and return collections of indices per slice.
    BeamBins bins;
    bins.build(
        np, particle_ptr, cbx,
        // Pass lambda function that returns the slice index
        [=] AMREX_GPU_HOST_DEVICE (const BeamParticleContainer::ParticleType& p)
        noexcept -> amrex::IntVect
        {
            return amrex::IntVect(
                AMREX_D_DECL(0, 0, static_cast<int>((p.pos(2)-plo[2])*dxi[2]-lo.z)));
        });

    return bins;
}
