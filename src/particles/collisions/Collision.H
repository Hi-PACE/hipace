#ifndef HIPACE_COLLISION_H_
#define HIPACE_COLLISION_H_

#include "particles/plasma/MultiPlasma.H"

#include <AMReX.H>
#include <AMReX_GpuBuffer.H>

#include <string>
#include <vector>

/**
 * \brief Electron-impact ionization between plasma species: a projectile electron species
 * ionizes a target species (ion or neutral atom), creating new electrons in a third species.
 * Pairing follows Perez et al. (2012) with the weight rescaling of Higginson et al. (2020);
 * the cross section is the RBEB model of Kim et al. (2000).
 */
class Collision
{
    std::string m_projectile_name = "";
    std::string m_target_name = "";
    std::string m_new_electron_name = "";
    std::string m_collision_type = "";
    std::string m_physical_element ="";
    int m_atomic_number = 0;
    amrex::Gpu::Buffer<amrex::Real> m_binding_energies;
    amrex::Gpu::Buffer<amrex::Real> m_ionization_energies;

public:

    /** Read parameters from the input file and check them against the plasma species
     *
     * \param[in] multi_plasma all plasma species (already initialized)
     * \param[in] collision_name name of this impact ionization process
     */
    void ReadParameters (
        const MultiPlasma& multi_plasma,
        std::string const collision_name);

    /** public wrapper function to handle the collision
     *
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     **/
    void doCollision (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma);

    /**
     * \brief Pair projectile and target particles cell by cell (random pairing as in
     * Perez et al. 2012) and apply collision_function to each pair; create the new
     * electrons with ionization_function
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     * \param[in] collision_function check if a collision between two particles produces a new electron,
     *  redistribute the energies and the momenta and update the old particle properties accordingly
     * \param[in] ionization_function initialize newly produced electrons
     **/
    template <class F, class G>
    void pairAndIonize (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        F const& collision_function,
        G const& ionization_function);

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
