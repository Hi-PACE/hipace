/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef MULTIPLASMA_H_
#define MULTIPLASMA_H_

#include "PlasmaParticleContainer.H"
#include "particles/sorting/TileSort.H"
#include "fields/Fields.H"
#include "particles/collisions/CoulombCollision.H"

class MultiPlasma
{

public:

    /** Constructor
     */
    MultiPlasma ();

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
     * \param[in] deposit_jx_jy if true, deposit to jx and jy
     * \param[in] deposit_jz if true, deposit to jz
     * \param[in] deposit_rho if true, deposit to rho
     * \param[in] deposit_chi if true, deposit chi
     * \param[in] deposit_rhomjz if true, deposit rhomjz
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void DepositCurrent (
        Fields & fields, int which_slice, bool deposit_jx_jy,
        bool deposit_jz, bool deposit_rho, bool deposit_chi, bool deposit_rhomjz,
        amrex::Vector<amrex::Geometry> const& gm, int const lev);

    /** Loop over plasma species and depose Sx and Sy into the current 2D slice in fields
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void ExplicitDeposition (Fields& fields, amrex::Vector<amrex::Geometry> const& gm,
                             int const lev);

    /** \brief Return max charge density, to compute the adaptive time step.
     *
     * the max is taken across species AND include m_adaptive_density, giving a way to
     * specify a density to the adaptive time step calculator even with no plasma species.
     */
    amrex::Real maxChargeDensity (amrex::Real z);

    /** \brief Gather field values and push particles
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] temp_slice if true, the temporary data (x_temp, ...) will be used
     * \param[in] lev MR level
     */
    void AdvanceParticles (
        const Fields & fields, amrex::Vector<amrex::Geometry> const& gm, bool temp_slice, int lev);

    /** \brief Loop over plasma species and deposit their neutralizing background, if needed
     *
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] which_slice slice in which the densities are deposited
     * \param[in] gm Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void DepositNeutralizingBackground (
        Fields & fields, int which_slice,
        amrex::Vector<amrex::Geometry> const& gm, int const lev);

    /** Calculates Ionization Probability and makes new Plasma Particles
     *
     * \param[in] lev MR level
     * \param[in] geom Geometry of the simulation, to get the cell size
     * \param[in] fields the general field class
     */
    void DoFieldIonization (const int lev, const amrex::Geometry& geom, const Fields& fields);

    bool IonizationOn () const;
    /** \brief whether any plasma species uses a neutralizing background, e.g. no ion motion */
    bool AnySpeciesNeutralizeBackground () const;

    /** \brief sort particles of all containers by tile logically, and store results in m_all_bins
     *
     * \param[in] bx transverse box on which the particles are sorted
     * \param[in] geom Geometry object
     */
    void TileSort (amrex::Box bx, amrex::Geometry geom);

    /** returns a Vector of names of the plasmas */
    const amrex::Vector<std::string>& GetNames() const {return m_names;}

    /** returns number of plasma species */
    int GetNPlasmas() const {return m_nplasmas;}

    /** Reorder particles to speed-up current deposition
     * \param[in] islice zeta slice index
     */
    void ReorderParticles (const int islice);

    /** \brief Store the finest level of every plasma particle in the cpu() attribute.
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] geom3D Geometry object for the whole domain
     * \param[in] to_prev if particles should be tagged to x_prev and y_prev
     */
    void TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                     const bool to_prev=false);

    /** Compute reduced plasma diagnostics of current slice, store in member variable.
     * \param[in] step time step of simulation
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] max_step maximum time step of simulation
     * \param[in] physical_time physical time at the given step
     * \param[in] max_time maximum time of simulation
     */
    void InSituComputeDiags (int step, int islice, int max_step,
                             amrex::Real physical_time, amrex::Real max_time);
    /** Write reduced beam diagnostics to file
     * \param[in] step time step of simulation
     * \param[in] time physical time at the given step
     * \param[in] geom Simulation geometry
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom,
                            int max_step, amrex::Real max_time);

    int m_sort_bin_size {32}; /**< Tile size to sort plasma particles */

    amrex::Vector<PlasmaParticleContainer> m_all_plasmas; /**< contains all plasma containers */
    int m_nplasmas = 0; /**< number of plasma containers */

    amrex::Vector<PlasmaBins> m_all_bins; /**< Logical tile bins for all plasma containers */
    amrex::Vector<std::string> m_names {"no_plasma"}; /**< names of all plasma containers */
private:
    /** Background (hypothetical) density, used to compute the adaptive time step */
    amrex::Real m_adaptive_density = 0.;

};

#endif // MULTIPLASMA_H_
