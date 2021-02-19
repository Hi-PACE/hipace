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
reorderParticlesBySlice (
    BeamParticleContainer& a_beam,
    const amrex::DenseBins<BeamParticleContainer::ParticleType>& a_bins)
{
    int const np = a_beam.numParticles();

    BeamParticleContainer tmp(a_beam.get_name());
    tmp.resize(np);

    amrex::gatherParticles(tmp, a_beam, np, a_bins.permutationPtr());

    a_beam.swap(tmp);
}

void
sortParticlesByBox (
    BeamParticleContainer& a_beam,
    const amrex::BoxArray a_ba, const amrex::Geometry& a_geom)
{
    amrex::ParticleLocator<amrex::DenseBins<amrex::Box> > particle_locator;
    particle_locator.build(a_ba, a_geom);
    auto assign_grid = particle_locator.getGridAssignor();

    int const np = a_beam.numParticles();
    BeamParticleContainer::ParticleType const* particle_ptr = a_beam.GetArrayOfStructs()().data();

    constexpr unsigned int max_unsigned_int = std::numeric_limits<unsigned int>::max();

    int num_boxes = a_ba.size();
    amrex::Gpu::DeviceVector<unsigned int> box_counts;
    amrex::Gpu::DeviceVector<unsigned int> box_offsets;
    amrex::Gpu::DeviceVector<unsigned int> dst_indices;

    box_counts.resize(num_boxes, 0);
    box_offsets.resize(num_boxes);
    dst_indices.resize(np);

    auto p_box_counts = box_counts.dataPtr();
    auto p_dst_indices = dst_indices.dataPtr();

    AMREX_FOR_1D ( np, i,
    {
        int dst_box = assign_grid(particle_ptr[i]);
        if (dst_box >= 0)  // what about ones that leave transversely?
        {
            unsigned int index = amrex::Gpu::Atomic::Inc(
                &p_box_counts[index], max_unsigned_int);
            p_dst_indices[i] = index;
        }
    });

    amrex::Gpu::exclusive_scan(box_counts.begin(), box_counts.end(), box_offsets.begin());

    BeamParticleContainer tmp(a_beam.get_name());
    tmp.resize(np);

    amrex::scatterParticles(tmp, a_beam, np, dst_indices.dataPtr());

    a_beam.swap(tmp);
}
