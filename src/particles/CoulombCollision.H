#ifndef HIPACE_COULOMB_COLLISION_H_
#define HIPACE_COULOMB_COLLISION_H_

#include "PlasmaParticleContainer.H"

#include <AMReX_DenseBins.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>

class CoulombCollision
{
public:
    int  m_species1_index;
    int  m_species2_index;
    bool m_isSameSpecies;
    amrex::Real m_CoulombLog;

    CoulombCollision(
        const std::vector<std::string>& species_names,
        std::string const collision_name);

    static void doCoulombCollision (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom, PlasmaParticleContainer& species1,
        PlasmaParticleContainer& species2, const bool is_same_species, const amrex::Real CoulombLog);

};

#endif // HIPACE_COULOMB_COLLISION_H_
