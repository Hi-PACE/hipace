#ifndef HIPACE_BeamParticleContainer_H_
#define HIPACE_BeamParticleContainer_H_

#include <AMReX_Particles.H>

struct BeamIdx
{
    enum {
        w = 0,      // weight
        ux, uy, uz, // momentum
        nattribs
    };
};

class BeamParticleContainer
    : public amrex::ParticleContainer<0, 0, BeamIdx::nattribs>
{};

#endif
