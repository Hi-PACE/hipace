#include "PlasmaParticleContainer.H"

PlasmaParticleContainer::PlasmaParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainer<0,0,PlasmaIdx::nattribs>(amr_core->GetParGDB())
{}

void
PlasmaParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();
    amrex::IntVect ppc {1,1,1};
    InitParticles(ppc,1.e-3,1.e-3,1.,geom);
}
