#ifndef HIPACE_MultiBeamParticleContainer_H_
#define HIPACE_MultiBeamParticleContainer_H_

#include <AMReX_AmrCore.H>

#include "BeamParticleContainer.H"

class MultiBeamParticleContainer
{
public:

    MultiBeamParticleContainer (amrex::AmrCore* amr_core);

    ~MultiBeamParticleContainer() {}

    BeamParticleContainer&
    GetParticleContainer (int ispecies) const {return *allcontainers[ispecies];}

    amrex::Vector<std::unique_ptr<BeamParticleContainer>> allcontainers;
};

#endif
