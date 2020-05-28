#include "BeamParticleContainer.H"

void
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();
    amrex::IntVect ppc {2,2,2};
    const amrex::RealBox& bounds = geom.ProbDomain();
    InitParticles(ppc,1.e-3,1.e-3,1.,geom,bounds);
}
