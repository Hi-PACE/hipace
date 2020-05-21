#include "MultiPlasmaParticleContainer.H"

MultiPlasmaParticleContainer::MultiPlasmaParticleContainer (AmrCore* amr_core)
{
    constexpr int nspecies = 1;
    allcontainers.resize(nspecies);
    for (int i = 0; i < nspecies; ++i)
    {
        allcontainers[i].reset(new PlasmaParticleContainer(amr_core);
    }
};
