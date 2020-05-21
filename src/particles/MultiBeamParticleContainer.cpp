#include "MultiBeamParticleContainer.H"

MultiBeamParticleContainer::MultiBeamParticleContainer (AmrCore* amr_core)
{
    constexpr int nbeams = 1;
    allcontainers.resize(nbeams);
    for (int i = 0; i < nspecies; ++i)
    {
        allcontainers[i].reset(new BeamParticleContainer(amr_core);
    }
};
