#include "MultiBeam.H"
#include "deposition/BeamDepositCurrent.H"
#include "particles/BinSort.H"
#include "pusher/BeamParticleAdvance.H"

MultiBeam::MultiBeam (amrex::AmrCore* amr_core)
{

    amrex::ParmParse pp("beams");
    pp.getarr("names", m_names);
    m_nbeams = m_names.size();
    for (int i = 0; i < m_nbeams; ++i) {
        m_all_beams.emplace_back(BeamParticleContainer(amr_core, m_names[i]));
    }
}

void
MultiBeam::InitData (const amrex::Geometry& geom)
{
    for (auto& beam : m_all_beams) {
        beam.InitData(geom);
    }
}

void
MultiBeam::DepositCurrent (Fields& fields, const amrex::Geometry& geom, const int lev)
{
    for (auto& beam : m_all_beams) {
        ::DepositCurrent(beam, fields, geom, lev);
    }
}

void
MultiBeam::DepositCurrentSlice (
    Fields& fields, const amrex::Geometry& geom, const int lev, int islice,
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins)
{
    for (int i=0; i<m_nbeams; i++) {
        ::DepositCurrentSlice(m_all_beams[i], fields, geom, lev, islice, bins[i]);
    }
}

amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>
MultiBeam::findParticlesInEachSlice (int lev, int ibox, amrex::Box bx, amrex::Geometry& geom)
{
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins;
    for (auto& beam : m_all_beams) {
        bins.emplace_back(::findParticlesInEachSlice(lev, ibox, bx, beam, geom));
    }
    return bins;
}

void
MultiBeam::Redistribute ()
{
    for (auto& beam : m_all_beams) {
        beam.Redistribute();
    }
}

void
MultiBeam::AdvanceBeamParticlesSlice (
    Fields& fields, amrex::Geometry const& gm, int const lev, const int islice,
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins)
{
    for (int i=0; i<m_nbeams; i++) {
        ::AdvanceBeamParticlesSlice(m_all_beams[i], fields, gm, lev, islice, bins[i]);
    }
}

void
MultiBeam::WritePlotFile (const std::string& filename)
{
    amrex::Vector<int> plot_flags(BeamIdx::nattribs, 1);
    amrex::Vector<int> int_flags(BeamIdx::nattribs, 1);
    amrex::Vector<std::string> real_names {"w","ux","uy","uz"};
    AMREX_ALWAYS_ASSERT(real_names.size() == BeamIdx::nattribs);
    amrex::Vector<std::string> int_names {};
    for (auto& beam : m_all_beams){
        beam.WritePlotFile(filename, beam.get_name(), plot_flags, int_flags, real_names, int_names);
    }
}

void
MultiBeam::NotifyNumParticles (MPI_Comm a_comm_z)
{
    for (auto& beam : m_all_beams) {
        beam.NotifyNumParticles(a_comm_z);
    }
}

void
MultiBeam::WaitNumParticles (MPI_Comm a_comm_z)
{
    for (auto& beam : m_all_beams) {
        beam.WaitNumParticles(a_comm_z);
    }
}

void
MultiBeam::ConvertUnits (ConvertDirection convert_direction)
{
    for (auto& beam : m_all_beams) {
        beam.ConvertUnits(convert_direction);
    }
}
