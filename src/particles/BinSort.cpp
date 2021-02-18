#include "BinSort.H"

#include <AMReX_ParticleTransformation.H>

amrex::DenseBins<BeamParticleContainer::ParticleType>
findParticlesInEachSlice (
    int /*lev*/, int /*ibox*/, amrex::Box bx,
    BeamParticleContainer& beam, amrex::Geometry& geom)
{
    // Slice box: only 1 cell transversally, same as bx longitudinally.
    const amrex::Box cbx ({0,0,bx.smallEnd(2)}, {0,0,bx.bigEnd(2)});

    // Extract particle structures for this tile
    int const np = beam.numParticles();
    BeamParticleContainer::ParticleType const* particle_ptr = beam.GetArrayOfStructs()().data();

    // Extract box properties
    const auto lo = lbound(cbx);
    const auto dxi = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();

    // Find the particles that are in each slice and return collections of indices per slice.
    amrex::DenseBins<BeamParticleContainer::ParticleType> bins;
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

void
reorderParticlesInEachSlice (
    BeamParticleContainer& a_beam,
    const amrex::DenseBins<BeamParticleContainer::ParticleType>& a_bins)
{
    const auto perm_ptr = a_bins.permutationPtr();
    int const np = a_beam.numParticles();

    BeamParticleContainer tmp("tmp");
    tmp.resize(np);

    amrex::scatterParticles(tmp, a_beam, np, perm_ptr);

    a_beam.swap(tmp);
}
