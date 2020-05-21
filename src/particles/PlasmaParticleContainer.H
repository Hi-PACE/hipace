#ifndef HIPACE_PlasmaParticleContainer_H_
#define HIPACE_PlasmaParticleContainer_H_

#include <AMReX_Particles.H>

struct PlasmaIdx
{
    enum {
        w = 0,                               // weight
        ux, uy,                              // momentum
        phi,                                 //
        Fx1, Fx2, Fx3, Fx4, Fx5,             //
        Fy1, Fy2, Fy3, Fy4, Fy5,             //
        Fux1, Fux2, Fux3, Fux4, Fux5,        //
        Fuy1, Fuy2, Fuy3, Fuy4, Fuy5,        //
        Fphi1, Fphi2, Fphi3, Fphi4, Fphi5,   //
        nattribs
    };
};

class PlasmaParticleContainer
    : public amrex::ParticleContainer<0, 0, PlasmaIdx::nattribs>
{};

#endif
