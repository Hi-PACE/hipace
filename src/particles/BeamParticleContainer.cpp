#include "BeamParticleContainer.H"

void
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();
    amrex::IntVect ppc {2,2,2};
    const amrex::RealBox bounds({AMREX_D_DECL(-5.0e-6,-5.e-6,5.e-6)},
                                {AMREX_D_DECL( 5.e-6, 5.e-6, 10.e-6)});
    // const amrex::RealBox& bounds = geom.ProbDomain();
    InitParticles(ppc,1.e-3,1.e-3,1.e3,geom,bounds);
}
