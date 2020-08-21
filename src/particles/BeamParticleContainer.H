#ifndef HIPACE_BeamParticleContainer_H_
#define HIPACE_BeamParticleContainer_H_

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

/** \brief Map names and indices for beam particles attributes (SoA data) */
struct BeamIdx
{
    enum {
        w = 0,      // weight
        ux, uy, uz, // momentum
        nattribs
    };
};

/** \brief Container for particles of 1 beam species. */
class BeamParticleContainer
    : public amrex::ParticleContainer<0, 0, BeamIdx::nattribs>
{
public:
    /** Constructor */
    explicit BeamParticleContainer (amrex::AmrCore* amr_core) :
        amrex::ParticleContainer<0,0,BeamIdx::nattribs>(amr_core->GetParGDB())
    {
        ReadParameters();
    }

    /** Read parameters in the input file */
    void ReadParameters ();

    /** Allocate data for the beam particles and initialize particles with requested beam profile
     */
    void InitData (const amrex::Geometry& geom);

    /** Initialize a can beam */
    void InitCanBeam (
        const amrex::IntVect&  a_num_particles_per_cell,
        const amrex::RealVect& a_thermal_momentum_std,
        const amrex::RealVect& a_thermal_momentum_mean,
        const amrex::Real      a_density,
        const amrex::Geometry& a_geom,
        const amrex::Real     a_zmin,
        const amrex::Real     a_zmax,
        const amrex::Real     a_radius);

private:
    amrex::Real m_zmin; /**< Min longitudinal position of the can beam */
    amrex::Real m_zmax; /**< Max longitudinal position of the can beam */
    amrex::Real m_radius; /**< Radius of the can beam */
    amrex::Real m_density; /**< Density of the can beam */
    amrex::IntVect m_ppc {1, 1, 1}; /**< Number of particles per cell in each direction */
    amrex::RealVect m_u_mean {0,0,0}; /**< Avg momentum in each direction normalized by m*c */
    amrex::RealVect m_u_std {0,0,0}; /**< Thermal momentum in each direction normalized by m*c */
};

/** \brief Iterator over boxes in a particle container */
class BeamParticleIterator : public amrex::ParIter<0,0,BeamIdx::nattribs>
{
public:
    using amrex::ParIter<0,0,BeamIdx::nattribs>::ParIter;
    /** Constructor */
    BeamParticleIterator (ContainerType& pc, int level): ParIter(pc, level) {}
};

#endif
