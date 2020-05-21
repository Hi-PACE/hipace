#ifndef HIPACE_MultiPlasmaParticleContainer_H_
#define HIPACE_MultiPlasmaParticleContainer_H_

#include <AMReX_AmrCore.H>

#include "PlasmaParticleContainer.H"

class MultiPlasmaParticleContainer
{
public:

    MultiPlasmaParticleContainer (amrex::AmrCore* amr_core);

    ~MultiPlasmaParticleContainer() {}

    PlasmaParticleContainer&
    GetParticleContainer (int ispecies) const {return *allcontainers[ispecies];}

    amrex::Vector<std::unique_ptr<PlasmaParticleContainer>> allcontainers;
};

#endif
