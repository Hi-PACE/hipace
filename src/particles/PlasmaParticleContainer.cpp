#include "PlasmaParticleContainer.H"

PlasmaParticleContainer::PlasmaParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainer<0,0,PlasmaIdx::nattribs>(amr_core->GetParGDB())
{}

void
PlasmaParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();
    const amrex::IntVect ppc {1,1,1};

    const int dir = 2;
    const amrex::Real dx = geom.CellSize(dir);
    const amrex::Real hi = geom.ProbHi(dir);
    const amrex::Real lo = hi - dx;

    amrex::RealBox particleBox = geom.ProbDomain();
    particleBox.setHi(dir, hi);
    particleBox.setLo(dir, lo);

    InitParticles(ppc,1.e-3,1.e-3,1.,geom,particleBox);
}
