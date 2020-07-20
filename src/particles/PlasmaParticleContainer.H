#ifndef HIPACE_PlasmaParticleContainer_H_
#define HIPACE_PlasmaParticleContainer_H_

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

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
{
public:
    explicit PlasmaParticleContainer (amrex::AmrCore* amr_core);

    void InitData (const amrex::Geometry& geom);

    void InitParticles (
        const amrex::IntVect&  a_num_particles_per_cell,
        const amrex::Real      a_thermal_momentum_std,
        const amrex::Real      a_thermal_momentum_mean,
        const amrex::Real      a_density,
        const amrex::Geometry& a_geom,
        const amrex::RealBox&  a_bounds);

    amrex::Real m_density; /**< Density of the plasma */
    amrex::IntVect m_ppc {1, 1, 1}; /**< Number of particles per cell in each direction */
    amrex::Real m_uz_mean = 0.; /**< Avg momentum along z, normalized by m*c */
    amrex::Real m_u_std = 0.; /**< Thermal momentum, normalized by m*c. Same in all directions */
};

class PlasmaParticleIterator : public amrex::ParIter<0,0,PlasmaIdx::nattribs>
{
public:
    using amrex::ParIter<0,0,PlasmaIdx::nattribs>::ParIter;
    PlasmaParticleIterator (ContainerType& pc, int level): ParIter(pc, level) {}
};

#endif
