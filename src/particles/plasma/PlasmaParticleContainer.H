/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_PlasmaParticleContainer_H_
#define HIPACE_PlasmaParticleContainer_H_

#include "fields/Fields.H"
#include "utils/Parser.H"
#include "utils/GPUUtil.H"
#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>
#include <map>

/** \brief Map names and indices for plasma particles attributes (SoA data) */
struct PlasmaIdx
{
    enum {
        x=0, y,             // position
        w,                  // weight, this will be returned by pos(2)
        ux, uy,             // momentum
        psi,                // pseudo-potential at the particle position. ATTENTION what is stored is actually normalized psi+1
        x_prev, y_prev,     // positions on the last non-temp slice

        ux_half_step,       // momentum half a step behind the current slice for leapfrog pusher
        uy_half_step,       // at the same step for AB5 pusher
        psi_half_step,      // never effected by temp slice
#ifdef HIPACE_USE_AB5_PUSH
        Fx1, Fx2, Fx3, Fx4, Fx5,             // AB5 force terms
        Fy1, Fy2, Fy3, Fy4, Fy5,             //
        Fux1, Fux2, Fux3, Fux4, Fux5,        //
        Fuy1, Fuy2, Fuy3, Fuy4, Fuy5,        //
        Fpsi1, Fpsi2, Fpsi3, Fpsi4, Fpsi5,   //
#endif
        real_nattribs
    };
    enum {
        ion_lev,            // ionization level
        int_nattribs
    };
};

/** \brief Container for particles of 1 plasma species. */
class PlasmaParticleContainer
    : public amrex::ParticleContainerPureSoA<PlasmaIdx::real_nattribs, PlasmaIdx::int_nattribs>
{
public:
    /** Constructor */
    explicit PlasmaParticleContainer (std::string name) :
        amrex::ParticleContainerPureSoA<PlasmaIdx::real_nattribs,PlasmaIdx::int_nattribs>(),
        m_name(name)
    {
        ReadParameters();
    }

    /** Read parameters from the input file
     */
    void ReadParameters ();

    /** Allocate data for the beam particles and initialize particles with requested beam profile
    * \param[in] geom Geometry object for the whole domain
     */
    void InitData (const amrex::Geometry& geom);

    /** Initialize 1 xy slice of particles, with fixed number of particles per cell
     *
     * \param[in] a_thermal_momentum_std standard deviation of the momentum distribution (3d)
     * \param[in] a_thermal_momentum_mean average momentum (3d)
     * \param[in] a_radius plasma radius. Only particles with x**2+y**2<a_radius**2 are injected
     * \param[in] a_hollow_core_radius hollow core plasma radius.
     *            Only particles with x**2+y**2>a_hollow_core_radius**2 are injected
     */
    void InitParticles (
        const amrex::RealVect& a_thermal_momentum_std,
        const amrex::RealVect& a_thermal_momentum_mean,
        const amrex::Real a_radius,
        const amrex::Real a_hollow_core_radius);

    /** Initialize ADK prefactors of ionizable plasmas
     *
     * \param[in] geom Geometry of the simulation, to get the cell size
     * \param[in] product_pc The electron plasma PC that this Ion Plasma ionizes to
     * \param[in] background_density_SI background plasma density (only needed for normalized units)
     */
    void InitIonizationModule (const amrex::Geometry& geom, PlasmaParticleContainer* product_pc,
                               const amrex::Real background_density_SI);

    /** Calculates Ionization Probability and generates new Plasma Particles
     *
     * \param[in] lev MR level
     * \param[in] geom Geometry of the simulation, to get the cell size
     * \param[in] fields the general field class
     * \param[in] background_density_SI background plasma density (only needed for normalized units)
     */
    void IonizationModule (const int lev,
                           const amrex::Geometry& geom,
                           const Fields& fields,
                           const amrex::Real background_density_SI);

    /** Reorder particles to speed-up current deposition
     * \param[in] islice zeta slice index
     */
    void ReorderParticles (const int islice);

    /** Update m_density_func with m_density_table if applicable
     * \param[in] pos_z z position to evaluate m_density_table
     */
    void UpdateDensityFunction (const amrex::Real pos_z);

    /** \brief Store the finest level of every plasma particle in the cpu() attribute.
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] geom3D Geometry object for the whole domain
     * \param[in] to_prev if particles should be tagged to x_prev and y_prev
     */
    void TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                     const bool to_prev=false);

    /** returns u_mean of the plasma distribution */
    amrex::RealVect GetUMean () const {return m_u_mean;}

    /** returns u_std of the plasma distribution */
    amrex::RealVect GetUStd () const {return m_u_std;}

    /** Returns elementary charge q_e (or -q_e for electrons).
     * TODO: for collisions, return the charge of the fully-ionized atom
     * (atomic number * elementary charge)
     */
    amrex::Real GetCharge () const {return m_charge;}

    /** Returns mass of physical species */
    amrex::Real GetMass () const {return m_mass;}

    /** Returns name of the plasma */
    const std::string& GetName () const {return m_name;}

    /** Compute in-situ plasma diagnostics of current slice, store in member variable
     * \param[in] islice current slice, on which diags are computed.
     */
    void InSituComputeDiags (int islice);
    /** Dump in-situ reduced diagnostics to file.
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] geom Geometry object for the whole domain
     */
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom);

    amrex::Parser m_parser; /**< owns data for m_density_func */
    amrex::ParserExecutor<3> m_density_func; /**< Density function for the plasma */
    amrex::Real m_min_density {0.}; /**< minimal density at which particles are injected */
    bool m_use_density_table; /**< if a density value table was specified */
    /** plasma density value table, key: position=c*time, value=density function string */
    std::map<amrex::Real, std::string> m_density_table;
    bool m_do_symmetrize = false; /**< Option to symmetrize the plasma */
    /** maximum weighting factor gamma/(Psi +1) before particle is regarded as violating
     *  the quasi-static approximation and is removed */
    amrex::Real m_max_qsa_weighting_factor {35.};
    amrex::Real m_radius {std::numeric_limits<amrex::Real>::infinity()}; /**< radius of the plasma */
    /** radius of the plasma insitu diagnostics */
    amrex::Real m_insitu_radius {std::numeric_limits<amrex::Real>::infinity()};
    amrex::Real m_hollow_core_radius {0.}; /**< hollow core radius of the plasma */
    bool m_use_fine_patch = false;
    int m_fine_transition_cells = 5;
    amrex::Parser m_parser_fine_patch; /**< owns data for m_fine_patch_func */
    amrex::ParserExecutor<2> m_fine_patch_func;
    amrex::Array<int, 2> m_ppc {0,0}; /**< Number of particles per cell in each direction */
    amrex::Array<int, 2> m_ppc_fine {0,0}; /**< Number of particles per cell in each direction */
    /**< Prevent a particle from being initialized at the domain center */
    bool m_prevent_centered_particle = false;
    amrex::RealVect m_u_mean {0,0,0}; /**< Avg momentum in each direction normalized by m*c */
    amrex::RealVect m_u_std {0,0,0}; /**< Thermal momentum in each direction normalized by m*c */
    amrex::Real m_temperature_in_ev {0.}; /**< Temperature of the plasma in eV */
    /** whether to add a neutralizing background of immobile particles of opposite charge */
    bool m_neutralize_background = true;
    amrex::Real m_mass = 0; /**< mass of each particle of this species */
    amrex::Real m_charge = 0; /**< charge of each particle of this species, per Ion level */
    int m_init_ion_lev = -1; /**< initial Ion level of each particle */
    int m_n_subcycles = 1; /**< number of subcycles in the plasma particle push */
    bool m_can_ionize = false; /**< whether this plasma can ionize */
    std::string m_product_name = ""; /**< name of Ionization product plasma */
    PlasmaParticleContainer* m_product_pc = nullptr; /**< Ionization product plasma */
    /** to calculate Ionization probability with ADK formula */
    amrex::Gpu::DeviceVector<amrex::Real> m_adk_prefactor;
    /** to calculate Ionization probability with ADK formula */
    amrex::Gpu::DeviceVector<amrex::Real> m_adk_exp_prefactor;
    /** to calculate Ionization probability with ADK formula */
    amrex::Gpu::DeviceVector<amrex::Real> m_adk_power;
    /** After how many slices the particles are reordered. 0: off */
    int m_reorder_period = 0;
    /** 2D reordering index type. 0: cell, 1: node, 2: both */
    amrex::IntVect m_reorder_idx_type = {0, 0, 0};
    /** How often the insitu plasma diagnostics should be computed and written
     * Default is 0, meaning no output */
    int m_insitu_period {0};
private:
    std::string m_name; /**< name of the species */
    int m_nslices; /**< number of z slices of the domain */
    /** Number of real plasma properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nrp = 14;
    /** Number of int plasma properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nip = 1;
    /** Per-slice real plasma properties */
    amrex::Vector<amrex::Real> m_insitu_rdata;
    /** Per-slice int plasma properties */
    amrex::Vector<int> m_insitu_idata;
    /** Sum of all per-slice real plasma properties */
    amrex::Vector<amrex::Real> m_insitu_sum_rdata;
    /** Sum of all per-slice int plasma properties */
    amrex::Vector<int> m_insitu_sum_idata;
    /** Prefix/path for the output files */
    std::string m_insitu_file_prefix = "diags/plasma_insitu";
};

/** \brief Iterator over boxes in a particle container */
class PlasmaParticleIterator : public amrex::ParIterSoA<PlasmaIdx::real_nattribs,PlasmaIdx::int_nattribs>
{
public:
    /** Constructor */
    PlasmaParticleIterator (ContainerType& pc)
        : amrex::ParIterSoA<PlasmaIdx::real_nattribs,PlasmaIdx::int_nattribs>(pc, 0, DfltMfi) {}
};

#endif
