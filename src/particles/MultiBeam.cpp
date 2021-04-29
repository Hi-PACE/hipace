#include "MultiBeam.H"
#include "deposition/BeamDepositCurrent.H"
#include "particles/BinSort.H"
#include "pusher/BeamParticleAdvance.H"

MultiBeam::MultiBeam (amrex::AmrCore* /*amr_core*/)
{

    amrex::ParmParse pp("beams");
    pp.getarr("names", m_names);
    if (m_names[0] == "no_beam") return;
    m_nbeams = m_names.size();
    for (int i = 0; i < m_nbeams; ++i) {
        m_all_beams.emplace_back(BeamParticleContainer(m_names[i]));
    }
    m_n_real_particles.resize(m_nbeams, 0);
}

void
MultiBeam::InitData (const amrex::Geometry& geom)
{
    for (auto& beam : m_all_beams) {
        beam.InitData(geom);
    }
}

void
MultiBeam::DepositCurrentSlice (
    Fields& fields, const amrex::Geometry& geom, const int lev, int islice, const amrex::Box bx,
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins,
    const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox,
    const bool do_beam_jx_jy_deposition, const int which_slice)

{
    for (int i=0; i<m_nbeams; i++) {
        ::DepositCurrentSlice(m_all_beams[i], fields, geom, lev, islice, bx,
                              a_box_sorter_vec[i].boxOffsetsPtr()[ibox], bins[i],
                              do_beam_jx_jy_deposition, which_slice);
    }
}

amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>
MultiBeam::findParticlesInEachSlice (int lev, int ibox, amrex::Box bx, amrex::Geometry& geom,
                                     const amrex::Vector<BoxSorter>& a_box_sorter_vec)
{
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins;
    for (int i=0; i<m_nbeams; i++) {
        bins.emplace_back(::findParticlesInEachSlice(lev, ibox, bx, m_all_beams[i], geom, a_box_sorter_vec[i]));
    }
    return bins;
}

void
MultiBeam::sortParticlesByBox (
            amrex::Vector<BoxSorter>& a_box_sorter_vec,
            const amrex::BoxArray a_ba, const amrex::Geometry& a_geom)
{
    a_box_sorter_vec.resize(m_nbeams);
    for (int i=0; i<m_nbeams; i++) {
        a_box_sorter_vec[i].sortParticlesByBox(m_all_beams[i], a_ba, a_geom);
    }
}

void
MultiBeam::AdvanceBeamParticlesSlice (
    Fields& fields, amrex::Geometry const& gm, int const lev, const int islice, const amrex::Box bx,
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins,
    const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox)
{
    for (int i=0; i<m_nbeams; i++) {
        ::AdvanceBeamParticlesSlice(m_all_beams[i], fields, gm, lev, islice, bx,
                                    a_box_sorter_vec[i].boxOffsetsPtr()[ibox], bins[i]);
    }
}

int
MultiBeam::NumRealComps ()
{
    int comps = 0;
    if (get_nbeams() > 0){
        comps = m_all_beams[0].NumRealComps();
        for (auto& beam : m_all_beams) {
            AMREX_ALWAYS_ASSERT(beam.NumRealComps() == comps);
        }
    }
    return comps;
}

int
MultiBeam::NumIntComps ()
{
    int comps = 0;
    if (get_nbeams() > 0){
        comps = m_all_beams[0].NumIntComps();
        for (auto& beam : m_all_beams) {
            AMREX_ALWAYS_ASSERT(beam.NumIntComps() == comps);
        }
    }
    return comps;
}

void
MultiBeam::StoreNRealParticles ()
{
    for (int i=0; i<m_nbeams; i++) {
        m_n_real_particles[i] = m_all_beams[i].numParticles();
    }
}

int
MultiBeam::NGhostParticles (int ibeam, amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins, amrex::Box bx)
{
    amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const * offsets = bins[ibeam].offsetsPtr();
    return offsets[bx.bigEnd(Direction::z)+1] - offsets[bx.bigEnd(Direction::z)];
}

void
MultiBeam::RemoveGhosts ()
{
    for (int i=0; i<m_nbeams; i++){
        m_all_beams[i].resize(m_n_real_particles[i]);
    }
}

void
MultiBeam::PrepareGhostSlice (int it, const amrex::Box& bx, const amrex::Vector<BoxSorter>& box_sorters, const amrex::Geometry& geom)
{
    constexpr int lev = 0;
    char * psend_buffer = nullptr;
    for (int ibeam=0; ibeam<m_nbeams; ibeam++){
        // sort particles in box it, effectively the one directly left of the current box
        amrex::DenseBins<BeamParticleContainer::ParticleType> bins =
            ::findParticlesInEachSlice(lev, it, bx, m_all_beams[ibeam], geom, box_sorters[ibeam]);

        // Get mapping of particles in the last slice of this box.
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const * offsets = bins.offsetsPtr();
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type* indices = bins.permutationPtr();
        const int cell_start = offsets[bx.bigEnd(Direction::z)];
        const int cell_stop  = offsets[bx.bigEnd(Direction::z)+1];
        const int nghost = cell_stop - cell_start;

        // Prepare memory to copy ghost particles and append them to the end of the array.
        // For convenience, we do it the same way as in Wait/Notify: pack into a buffer,
        // unpack in the position of ghost particles.
        const amrex::Long psize = sizeof(BeamParticleContainer::SuperParticleType);
        const amrex::Long buffer_size = psize*nghost;
        psend_buffer = (char*) amrex::The_Pinned_Arena()->alloc(buffer_size);
        const amrex::Gpu::DeviceVector<int> comm_real(NumRealComps(), 1);
        const amrex::Gpu::DeviceVector<int> comm_int (NumIntComps(),  1);
        const auto p_comm_real = comm_real.data();
        const auto p_comm_int = comm_int.data();

        {
            auto& ptile = getBeam(ibeam);
            auto ptd = ptile.getParticleTileData();

            // Copy from last slice to buffer
            for (int i = 0; i < nghost; ++i) {
                ptd.packParticleData(psend_buffer, indices[cell_start+i], i*psize, p_comm_real, p_comm_int);
            }
        }

        {
            auto& ptile = getBeam(ibeam);
            auto old_size = ptile.numParticles();
            auto new_size = old_size + nghost;
            // Allocate memory in the particle array for the ghost particles
            ptile.resize(new_size);
            auto ptd = ptile.getParticleTileData();

            // Copy from buffer to ghost particles
            for (int i = 0; i < nghost; ++i) {
                ptd.unpackParticleData(
                    psend_buffer, i*psize, i+old_size, p_comm_real, p_comm_int);
            }
        }
    }
}
