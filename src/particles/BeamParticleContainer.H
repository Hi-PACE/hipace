#ifndef HIPACE_BeamParticleContainer_H_
#define HIPACE_BeamParticleContainer_H_

#include "profiles/GetInitialDensity.H"
#include "profiles/GetInitialMomentum.H"
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
    void InitBeam (
        const amrex::IntVect&  a_num_particles_per_cell,
        const GetInitialDensity& get_density,
        const GetInitialMomentum& get_momentum,
        const amrex::Geometry& a_geom,
        const amrex::Real     a_zmin,
        const amrex::Real     a_zmax,
        const amrex::Real     a_radius);

private:
    amrex::Real m_zmin; /**< Min longitudinal position of the can beam */
    amrex::Real m_zmax; /**< Max longitudinal position of the can beam */
    amrex::Real m_radius; /**< Radius of the can beam */
    amrex::IntVect m_ppc {1, 1, 1}; /**< Number of particles per cell in each direction */
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
