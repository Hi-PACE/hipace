#include "MultiBeam.H"
#include "deposition/BeamDepositCurrent.H"
#include "particles/BinSort.H"
#include "pusher/BeamParticleAdvance.H"
#include "pusher/GetAndSetPosition.H"

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
        const int nghost = m_all_beams[i].numParticles() - m_n_real_particles[i];
        ::DepositCurrentSlice(m_all_beams[i], fields, geom, lev, islice, bx,
                              a_box_sorter_vec[i].boxOffsetsPtr()[ibox], bins[i],
                              do_beam_jx_jy_deposition, which_slice, nghost);
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
    amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const * offsets = 0;
    offsets = bins[ibeam].offsetsPtr();
    return offsets[bx.bigEnd(Direction::z)+1-bx.smallEnd(Direction::z)]
        - offsets[bx.bigEnd(Direction::z)-bx.smallEnd(Direction::z)];
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
    for (int ibeam=0; ibeam<m_nbeams; ibeam++){

        const int offset_box_left = box_sorters[ibeam].boxOffsetsPtr()[it];
        const int offset_box_curr = box_sorters[ibeam].boxOffsetsPtr()[it+1];
        const int nghost = offset_box_curr - offset_box_left;

        // Resize particle array
        auto& ptile = getBeam(ibeam);
        int old_size = ptile.numParticles();
        auto new_size = old_size + nghost;
        ptile.resize(new_size);

        // Copy particles in box it to ghost particles
        // Access AoS particle data
        auto& aos = ptile.GetArrayOfStructs();
        const auto& pos_structs_src = aos.begin() + offset_box_left;
        const auto& pos_structs_dst = aos.begin() + old_size;
        // Access SoA particle data
        auto& soa = ptile.GetStructOfArrays(); // For momenta and weights
        const auto  wp_src = soa.GetRealData(BeamIdx::w).data()  + offset_box_left;
        const auto uxp_src = soa.GetRealData(BeamIdx::ux).data() + offset_box_left;
        const auto uyp_src = soa.GetRealData(BeamIdx::uy).data() + offset_box_left;
        const auto uzp_src = soa.GetRealData(BeamIdx::uz).data() + offset_box_left;
        const auto  wp_dst = soa.GetRealData(BeamIdx::w).data()  + old_size;
        const auto uxp_dst = soa.GetRealData(BeamIdx::ux).data() + old_size;
        const auto uyp_dst = soa.GetRealData(BeamIdx::uy).data() + old_size;
        const auto uzp_dst = soa.GetRealData(BeamIdx::uz).data() + old_size;

        amrex::ParallelFor(
            nghost,
            [=] AMREX_GPU_DEVICE (long idx) {
                pos_structs_dst[idx].id() = pos_structs_src[idx].id();
                pos_structs_dst[idx].pos(0) = pos_structs_src[idx].pos(0);
                pos_structs_dst[idx].pos(1) = pos_structs_src[idx].pos(1);
                pos_structs_dst[idx].pos(2) = pos_structs_src[idx].pos(2);
                wp_dst[idx] = wp_src[idx];
                uxp_dst[idx] = uxp_src[idx];
                uyp_dst[idx] = uyp_src[idx];
                uzp_dst[idx] = uzp_src[idx];
            }
            );
    }
}
