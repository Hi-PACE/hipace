#include "BinSort.H"

ParticleBins
findParticlesInEachCell( int const lev, amrex::Box bx,
                         ParticleTileType const& ptile, amrex::Geometry geom)
{
    // Slice box: only 1 cell transversally, same as bx longitudinally.
    const amrex::Box cbx ({0,0,bx.smallEnd(2)},{0,0,bx.bigEnd(2)});
    
    // Extract particle structures for this tile
    int const np = ptile.numParticles();
    ParticleType const* particle_ptr = ptile.GetArrayOfStructs()().data();

    // Extract box properties
    const auto lo = lbound(cbx);
    const auto dxi = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();

    // Find particles that are in each cell;
    // results are stored in the object `bins`.
    ParticleBins bins;
    bins.build(
        np, particle_ptr, cbx,
        // Pass lambda function that returns the cell index
        [=] AMREX_GPU_HOST_DEVICE (const ParticleType& p) noexcept -> amrex::IntVect
        {
            return amrex::IntVect(
                AMREX_D_DECL(0, 0, static_cast<int>((p.pos(2)-plo[2])*dxi[2]-lo.z)));
        });
    
    return bins;
}
