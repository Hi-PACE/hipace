#ifndef HIPACE_COLLISION_H_
#define HIPACE_COLLISION_H_

#include "particles/plasma/MultiPlasma.H"

#include <AMReX.H>

#include <string>

/**
 * \brief This class handles inelastic Coulomb collisions between 2 plasma species,
 * and eventually leads to an ionization event.
 */
class Collision
{
    std::string m_projectile_name = "";
    std::string m_target_name = "";
    std::string m_new_electron_name = "";
    std::string m_collision_type = "";
    std::string m_physical_element ="";
    bool m_has_collision_product = false;
    amrex::Gpu::Buffer<amrex::Real> m_binding_energies;
    amrex::Gpu::Buffer<amrex::Real> m_ionization_energies;

public:

    /** Read parameters from the input file */
    void ReadParameters (
        const std::vector<std::string>& plasma_species_names,
        std::string const collision_name);

    /**
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     **/
    void doCollision (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma);

    /**
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     * \param[in] collision_function return if a pair of particles collides
     * \param[in] ionizaiton_function initialize product particle
     **/
    template <class F, class G>
    void doCollisionImp (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        F const& collision_function,
        G const& ionizaiton_function);

    /**
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     **/
    void doElectronImpact (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma);
};

#endif // HIPACE_COLLISION_H_
