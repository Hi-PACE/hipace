#ifndef HIPACE_MultiBeamParticleContainer_H_
#define HIPACE_MultiBeamParticleContainer_H_

#include "BeamParticleContainer.H"

#include <AMReX_AmrCore.H>

class MultiBeamParticleContainer
{
public:

    MultiBeamParticleContainer (amrex::AmrCore* amr_core);

    ~MultiBeamParticleContainer() {}

    BeamParticleContainer&
    GetParticleContainer (int ispecies) const {return *allcontainers[ispecies];}

    amrex::Vector<std::unique_ptr<BeamParticleContainer>> allcontainers;

    void InitData (amrex::Geometry geom);

    void WritePlotFile (std::string filename);
};

#endif
