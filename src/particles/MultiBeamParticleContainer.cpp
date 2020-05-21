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
