#ifndef HIPACE_COLLISION_H_
#define HIPACE_COLLISION_H_

#include "particles/plasma/MultiPlasma.H"

#include <AMReX.H>

/**
 * \brief This class handles Coulomb collisions between 2 particle species
 * (can be plasma-plasma or beam-plasma, the species can be the same)
 */
class Collision
{
public:
    int m_inout_species1_index = -1;
    int m_inout_species2_index = -1;
    int m_out_species3_index = -1;
    std::string m_collision_type = "";

    /** Read parameters from the input file */
    void ReadParameters (
        const std::vector<std::string>& plasma_species_names,
        std::string const collision_name);

    /**
     * \param[in] lev MR level
     * \param[in] bx transverse box (plasma particles will be sorted per-cell on this box)
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     **/
    static void doCollision (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma);
};

#endif // HIPACE_COLLISION_H_
