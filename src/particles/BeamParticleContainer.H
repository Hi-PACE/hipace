#ifndef HIPACE_BeamParticleContainer_H_
#define HIPACE_BeamParticleContainer_H_

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

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
{
public:
    explicit BeamParticleContainer (amrex::AmrCore* amr_core) :
        amrex::ParticleContainer<0,0,BeamIdx::nattribs>(amr_core->GetParGDB())
    {}

    void InitData (const amrex::Geometry& geom);

    void InitParticles(
        const amrex::IntVect&  a_num_particles_per_cell,
        const amrex::Real      a_thermal_momentum_std,
        const amrex::Real      a_thermal_momentum_mean,
        const amrex::Real      a_density,
        const amrex::Geometry& a_geom,
        const amrex::RealBox&  a_bounds);

};


class BeamParticleIterator : public amrex::ParIter<0,0,BeamIdx::nattribs>
{
public:
    using amrex::ParIter<0,0,PIdx::nattribs>::ParIter;
    BeamParticleIterator (ContainerType& pc, int level): ParIter(pc, level) {};
};

#endif
