#include "MultiBeamParticleContainer.H"

MultiBeamParticleContainer::MultiBeamParticleContainer (amrex::AmrCore* amr_core)
{
    constexpr int nbeams = 1;
    allcontainers.resize(nbeams);
    for (int i = 0; i < nbeams; ++i)
    {
        allcontainers[i].reset(new BeamParticleContainer(amr_core));
    }
};

void
MultiBeamParticleContainer::InitData (amrex::Geometry geom)
{
    for (auto& pc : allcontainers){
        pc->InitData(geom);
    }
}

void
MultiBeamParticleContainer::WritePlotFile (std::string filename)
{
    amrex::Vector<int> plot_flags(BeamIdx::nattribs, 1);
    amrex::Vector<int> int_flags(BeamIdx::nattribs, 1);
    amrex::Vector<std::string> real_names {"w","ux","uy","uz"};
    AMREX_ALWAYS_ASSERT(real_names.size() == BeamIdx::nattribs);
    amrex::Vector<std::string> int_names {};
        
    for (auto& pc : allcontainers){
        // pc->WriteHeader(os);
        pc->WritePlotFile(
            filename, "species1",
            plot_flags, int_flags,
            real_names, int_names);
    }    
}
