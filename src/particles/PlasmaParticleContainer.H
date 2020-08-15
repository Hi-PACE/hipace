#ifndef HIPACE_PlasmaParticleContainer_H_
#define HIPACE_PlasmaParticleContainer_H_

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

/** \brief Map names and indices for plasma particles attributes (SoA data) */
struct PlasmaIdx
{
    enum {
        w = 0,                               // weight
        ux, uy,                              // momentum
        psi,                                 //
        Fx1, Fx2, Fx3, Fx4, Fx5,             //
        Fy1, Fy2, Fy3, Fy4, Fy5,             //
        Fux1, Fux2, Fux3, Fux4, Fux5,        //
        Fuy1, Fuy2, Fuy3, Fuy4, Fuy5,        //
        Fpsi1, Fpsi2, Fpsi3, Fpsi4, Fpsi5,   //
        nattribs
    };
};

/** \brief Container for particles of 1 plasma species. */
class PlasmaParticleContainer
    : public amrex::ParticleContainer<0, 0, PlasmaIdx::nattribs>
{
public:
    explicit PlasmaParticleContainer (amrex::AmrCore* amr_core);

    /** Allocate data for the beam particles and initialize particles with requested beam profile
     */
    void InitData (const amrex::Geometry& geom);

    /** Initialize 1 xy slice of particles, with fixed number of particles per cell */
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

/** \brief Iterator over boxes in a particle container */
class PlasmaParticleIterator : public amrex::ParIter<0,0,PlasmaIdx::nattribs>
{
public:
    using amrex::ParIter<0,0,PlasmaIdx::nattribs>::ParIter;
    PlasmaParticleIterator (ContainerType& pc, int level): ParIter(pc, level) {}
};

#endif
