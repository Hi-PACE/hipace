#ifndef HIPACE_COULOMB_COLLISION_H_
#define HIPACE_COULOMB_COLLISION_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "particles/beam/BeamParticleContainer.H"

#include <AMReX_DenseBins.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>

/**
 * \brief This class handles Coulomb collisions between 2 particle species
 * (can be plasma-plasma or beam-plasma, the species can be the same)
 */
class CoulombCollision
{
public:
    int  m_species1_index {-1};
    int  m_species2_index {-1};
    int  m_nbeams {0};
    bool m_isSameSpecies {false};
    amrex::Real m_CoulombLog {-1.};

    /** Constructor */
    CoulombCollision(
        const std::vector<std::string>& plasma_species_names,
        const std::vector<std::string>& beam_species_names,
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
     * \param[in] background_density_SI background plasma density (only needed for normalized units)
     **/
    static void doPlasmaPlasmaCoulombCollision (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom, PlasmaParticleContainer& species1,
        PlasmaParticleContainer& species2, bool is_same_species, amrex::Real CoulombLog,
        amrex::Real background_density_SI);

    /**
     * \brief Perform Coulomb collisions of a beam with a plasma species over a push by one beam time step
     *        Particles of both species are sorted per cell, paired, and collided pairwise.
     *
     * \param[in] lev MR level
     * \param[in] bx transverse box (plasma particles will be sorted per-cell on this box)
     * \param[in] geom corresponding geometry object
     * \param[in] islice_local local slice on which collisions are calculated
     * \param[in,out] species1 beam species
     * \param[in,out] species2 plasma species
     * \param[in] CoulombLog Value of the Coulomb logarithm used for the collisions. If <0, the
     *            Coulomb logarithm is deduced from the plasma temperature, measured in each cell.
     * \param[in] background_density_SI background plasma density (only needed for normalized units)
     **/
    static void doBeamPlasmaCoulombCollision (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom, int islice_local,
        BeamParticleContainer& species1, PlasmaParticleContainer& species2, amrex::Real CoulombLog,
        amrex::Real background_density_SI);

};

#endif // HIPACE_COULOMB_COLLISION_H_
