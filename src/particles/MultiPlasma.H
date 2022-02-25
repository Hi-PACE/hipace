#ifndef MULTIPLASMA_H_
#define MULTIPLASMA_H_

#include "PlasmaParticleContainer.H"
#include "TileSort.H"
#include "fields/Fields.H"
#include "CoulombCollision.H"

class MultiPlasma
{

public:

    /** Constructor
     * \param[in] amr_core AmrCore object
     */
    MultiPlasma (amrex::AmrCore* amr_core);

    /** Destructor */
    ~MultiPlasma () {}

    /** \brief Loop over plasma species and initialize them.
     *
     * \param[in] slice_ba slice boxarray, on which plasma particles are defined
     * \param[in] slice_dm DistributionMapping of the transverse slice domain
     * \param[in] slice_gm slice geometry
     * \param[in] gm Geometry of the simulation, to get the cell size
     */
    void InitData (amrex::Vector<amrex::BoxArray> slice_ba,
                   amrex::Vector<amrex::DistributionMapping> slice_dm,
                   amrex::Vector<amrex::Geometry> slice_gm, amrex::Vector<amrex::Geometry> gm);


    /** Loop over plasma species and depose their currents into the current 2D slice in fields
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] which_slice defines if this or the next slice is handled
     * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
     * \param[in] deposit_jx_jy if true, deposit to jx and jy
     * \param[in] deposit_jz if true, deposit to jz
     * \param[in] deposit_rho if true, deposit to rho
     * \param[in] deposit_j_squared if true, deposit jxx, jxy, jyy
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void DepositCurrent (
        Fields & fields, int which_slice, bool temp_slice, bool deposit_jx_jy, bool deposit_jz,
        bool deposit_rho, bool deposit_j_squared, amrex::Geometry const& gm, int const lev);

    /** \brief Return max density, to compute the adaptive time step.
     *
     * the max is taken across species AND include m_adaptive_density, giving a way to
     * specify a density to the adaptive time step calculator even with no plasma species.
     */
    amrex::Real maxDensity () const;

    /** \brief Loop over plasma species and Gather fields, update forces and push particles
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] temp_slice if true, the temporary data (x_temp, ...) will be used
     * \param[in] do_push boolean to define if plasma particles are pushed
     * \param[in] do_update boolean to define if the force terms are updated
     * \param[in] do_shift boolean to define if the force terms are shifted
     * \param[in] lev MR level
     */
    void AdvanceParticles (
        Fields & fields, amrex::Geometry const& gm, bool temp_slice, bool do_push,
        bool do_update, bool do_shift, int lev);

    /** \brief Resets the particle position x, y, to x_prev, y_prev
     *
     * \param[in] lev MR level
     * \param[in] initial whether this is initial reset at each time step. iF so, reset everything.
     */
    void ResetParticles (int lev, bool initial=false);

    /** \brief Loop over plasma species and deposit their neutralizing background, if needed
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] which_slice slice in which the densities are deposited
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] nlev number of MR levels
     */
    void DepositNeutralizingBackground (
        Fields & fields, int which_slice, amrex::Geometry const& gm, int const nlev);

    /** Calculates Ionization Probability and makes new Plasma Particles
     *
     * \param[in] lev MR level
     * \param[in] geom Geometry of the simulation, to get the cell size
     * \param[in] fields the general field class
     */
    void DoFieldIonization (const int lev, const amrex::Geometry& geom, Fields& fields);

    bool IonizationOn () const;
    /** \brief whether all plasma species use a neutralizing background, e.g. no ion motion */
    bool AllSpeciesNeutralizeBackground () const;

    /** \brief sort particles of all containers by tile logically, and store results in m_all_bins
     *
     * \param[in] bx transverse box on which the particles are sorted
     * \param[in] geom Geometry object
     */
    void TileSort (amrex::Box bx, amrex::Geometry geom);

    /** returns u_std of the plasma distribution */
    amrex::RealVect GetUStd () const;

    /** Return the nominal density */
    amrex::Real GetNominalDensity () const {return m_nominal_density;};

    /** Check whether the on-axis density is close enough to the nominal density,
     * to avoid machine-precision problems. */
    void CheckDensity () const;

    /** Perform binary elastic Coulomb collision, inter- and/or intra-species.
     *
     * The algorithm implemented is that of [Perez et al., Phys. Plasmas 19, 083104 (2012)]
     *
     * \param[in] lev MR level
     * \param[in] bx box on which plasma particles are sorted per-cell and collide together
     * \param[in] geom Corresponding gemetry
     */
    void doCoulombCollision (int lev, amrex::Box bx, amrex::Geometry geom);

    int m_sort_bin_size {32}; /**< Tile size to sort plasma particles */

    /** Number of binary collisions */
    int m_ncollisions = 0;

private:

    amrex::Vector<PlasmaParticleContainer> m_all_plasmas; /**< contains all plasma containers */
    amrex::Vector<PlasmaBins> m_all_bins; /**< Logical tile bins for all plasma containers */
    amrex::Vector<std::string> m_names; /**< names of all plasma containers */
    int m_nplasmas = 0; /**< number of plasma containers */
    /** Background (hypothetical) density, used to compute the adaptive time step */
    amrex::Real m_adaptive_density = 0.;
    amrex::Real m_nominal_density; /**< Density to normalize quantities in the explicit solver */

    /** User-input names of the binary collisions to be used */
    std::vector<std::string> m_collision_names;
    /** Vector of binary collisions */
    amrex::Vector< CoulombCollision > m_all_collisions;
};

#endif // MULTIPLASMA_H_
