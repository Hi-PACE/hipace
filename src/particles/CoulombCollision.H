#ifndef HIPACE_COULOMB_COLLISION_H_
#define HIPACE_COULOMB_COLLISION_H_

#include "PlasmaParticleContainer.H"

#include <AMReX_DenseBins.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>

/**
 * \brief This class handles Coulomb collisions between 2 plasma species (can be the same).
 */
class CoulombCollision
{
public:
    int  m_species1_index;
    int  m_species2_index;
    bool m_isSameSpecies;
    amrex::Real m_CoulombLog;

    /** Constructor */
    CoulombCollision(
        const std::vector<std::string>& species_names,
        std::string const collision_name);

    /**
     * \brief Perform Coulomb collisions of plasma species over longitudinal push by 1 cell.
     *        Particles of both species are sorted per cell, paired, and collided pairwise.
     *
     * \param[in] lev MR level
     * \param[in] bx transverse box (plasma particles will be sorted per-cell on this box)
     * \param[in] geom corresponding geometry object
     * \param[in,out] species1 first plasma species
     * \param[in,out] species2 second plasma species
     * \param[in] is_same_species whether both species are the same (intra-species collisions)
     * \param[in] CoulombLog Value of the Coulomb logarithm used for the collisions. If <0, the
     *            Coulomb logarithm is deduced from the plasma temperature, measured in each cell.
     **/
    static void doCoulombCollision (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom, PlasmaParticleContainer& species1,
        PlasmaParticleContainer& species2, const bool is_same_species, const amrex::Real CoulombLog);

};

#endif // HIPACE_COULOMB_COLLISION_H_
