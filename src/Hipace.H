/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Remi Lehe, Severin Diederichs, WeiqunZhang
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_H_
#define HIPACE_H_

#include "fields/Fields.H"
#include "fields/fft_poisson_solver/FFTPoissonSolver.H"
#include "particles/plasma/MultiPlasma.H"
#include "particles/beam/MultiBeam.H"
#include "particles/beam/BeamParticleContainer.H"
#include "utils/AdaptiveTimeStep.H"
#include "utils/GridCurrent.H"
#include "laser/MultiLaser.H"
#include "utils/Constants.H"
#include "utils/Parser.H"
#include "utils/MultiBuffer.H"
#include "diagnostics/Diagnostic.H"
#include "diagnostics/OpenPMDWriter.H"

#include <AMReX_AmrCore.H>
#ifdef AMREX_USE_LINEAR_SOLVERS
#  include <AMReX_MLALaplacian.H>
#  include <AMReX_MLMG.H>
#endif

#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <memory>

namespace hpmg { class MultiGrid; }

/** \brief Helper struct to initialize m_phys_const and Parser before amrex::AmrCore */
struct Hipace_early_init
{
    /** Constructor of m_phys_const, initialize Hipace::m_instance, Hipace::m_normalized_units
     * and Parser Constants */
    Hipace_early_init (Hipace* instance);

    /** Struct containing physical constants (which values depends on the unit system, determined
     * at runtime): SI or normalized units. */
    PhysConst m_phys_const;

    /** Order of the field gather and current deposition shape factor in the transverse directions
     */
    inline static int m_depos_order_xy = 2;
    /** Order of the field gather and current deposition shape factor in the longitudinal direction
     */
    inline static int m_depos_order_z = 0;
    /** Type of derivative used in explicit deposition. 0: analytic, 1: nodal, 2: centered
     */
    inline static int m_depos_derivative_type = 2;
    /* if the loop over depos_order is included in the loop over particles
     */
    inline static bool m_outer_depos_loop = false;

    /* Number of mesh refinement levels */
    int m_N_level = 1;
};

/** \brief Singleton class, that intialize, runs and finalizes the simulation */
class Hipace final : public Hipace_early_init
{
public:
    /** Ctor: read general input parameters, call constructors of main member variables
     * and initialize longitudinal and transverse MPI communicators */
    Hipace ();

    /** Get singleton instance */
    static Hipace& GetInstance ();

    /** Init AmrCore and allocate beam and plasma containers */
    void InitData ();

    /** Run the simulation. This function contains the loop over time steps */
    void Evolve ();

    /** Make Geometry, DistributionMapping and BoxArray for all MR levels */
    void MakeGeometry ();

    /** \brief Dump simulation data to file
     *
     * \param[in] output_step current iteration
     * \param[in] call_type whether the beams or the fields should be written to file
     */
    void WriteDiagnostics (int output_step, const OpenPMDWriterCallType call_type);

    /** \brief Return a copy of member struct for physical constants */
    PhysConst get_phys_const () {return m_phys_const;}

    /** \brief Full evolve on 1 slice
     *
     * \param[in] islice slice number
     * \param[in] step current time step
     */
    void SolveOneSlice (int islice, int step);

    /**
     * \brief Initialize Sx and Sy with the beam contributions
     *
     * \param[lev] MR level
     */
    void InitializeSxSyWithBeam (const int lev);

    /**
     * \brief Knowing the sources Sx, Sy and chi, apply MG to solve for Bx, By
     *
     * \param[lev] MR level
     * \param[in] which_slice defines if this or the salame slice is handled
     */
    void ExplicitMGSolveBxBy (const int lev, const int which_slice);

    /** \brief Reset plasma and field slice quantities to initial value.
     *
     * Typically done at the beginning of each iteration.
     */
    void ResetAllQuantities ();

    /**
     * \brief does Coulomb collisions between plasmas and beams
     */
    void doCoulombCollision ();

    /**
       \brief Returns the rank id of the head rank.
     */
    static int HeadRankID ()
    {
        return 0;
    }

    /**
       \brief Returns true on the head rank, otherwise false.
     */
    static bool HeadRank ()
    {
        return amrex::ParallelDescriptor::MyProc() == HeadRankID();
    }

    /** Version of the HiPACE executable
     *
     * @return detailed version string
     */
    static std::string Version ();

    /** 3D Geometry, vector over MR levels */
    amrex::Vector<amrex::Geometry> m_3D_geom;
    /** 3D DistributionMapping. Does not represent the parallelization pipeline correctly */
    amrex::Vector<amrex::DistributionMapping> m_3D_dm;
    /** 3D BoxArray, vector over MR levels. One box per rank */
    amrex::Vector<amrex::BoxArray> m_3D_ba;
    /** xy slice Geometry, vector over MR levels */
    amrex::Vector<amrex::Geometry> m_slice_geom;
    /** xy slice DistributionMapping, vector over MR levels */
    amrex::Vector<amrex::DistributionMapping> m_slice_dm;
    /** xy slice BoxArray, vector over MR levels. Contains only one box */
    amrex::Vector<amrex::BoxArray> m_slice_ba;
    /** Pointer to current (and only) instance of class Hipace */
    inline static Hipace* m_instance = nullptr;
    /** Whether to use normalized units */
    inline static bool m_normalized_units = false;
    /** All field data (3D array, slices) and field methods */
    Fields m_fields;
    /** Contains all beam species */
    MultiBeam m_multi_beam;
    /** Contains all plasma species */
    MultiPlasma m_multi_plasma;

    MultiBuffer m_multi_buffer;
    /** Number of processors in the longitudinal z direction */
    inline static int m_numprocs = 0;
    /** Number of time iterations */
    inline static int m_max_step = 0;
    /** Maximum simulation time */
    inline static amrex::Real m_max_time = std::numeric_limits<amrex::Real>::infinity();
    /** Physical time of the simulation. At the end of the time step, it is the physical time
     * at which the fields have been calculated. The beam is one step ahead. */
    inline static amrex::Real m_physical_time = 0.0;
    /** Physical time at the beginning of the simulation */
    inline static amrex::Real m_initial_time = 0.0;

    bool m_has_last_step = false;
    /** Level of verbosity */
    inline static int m_verbose = 0;
    /** Relative transverse B field error tolerance in the predictor corrector loop
     */
    inline static amrex::Real m_predcorr_B_error_tolerance = 4e-2;
    /** Maximum number of iterations in the predictor corrector loop
     */
    inline static int m_predcorr_max_iterations = 30;
    /** Average number of iterations in the predictor corrector loop
     */
    amrex::Real m_predcorr_avg_iterations = 0.;
    /** Average transverse B field error in the predictor corrector loop
     */
    amrex::Real m_predcorr_avg_B_error = 0.;
    /** Mixing factor between the transverse B field iterations in the predictor corrector loop
     */
    inline static amrex::Real m_predcorr_B_mixing_factor = 0.05;
    /** Whether the beams deposit Jx and Jy */
    inline static bool m_do_beam_jx_jy_deposition = true;
    /** Whether the jz-c*rho contribution of the beam is computed and used. If not, jz-c*rho=0 is assumed */
    inline static bool m_do_beam_jz_minus_rho = false;
    /** Whether to deposit rho (plasma) for diagnostics */
    inline static bool m_deposit_rho = false;
    /** Whether to deposit rho for every individual plasma for diagnostics */
    inline static bool m_deposit_rho_individual = false;
    /** Whether to interpolate the neutralizing background to MR levels 1 and 2 instead of depositing it */
    inline static bool m_interpolate_neutralizing_background = false;
    /** Whether to use tiling for particle operations */
#ifdef AMREX_USE_GPU
    inline static bool m_do_tiling = false;
#else
    inline static bool m_do_tiling = true;
#endif
    /** Whether the explicit field solver is used */
    inline static bool m_explicit = true;
    /** Relative tolerance for the multigrid solver, when using the explicit solver */
    inline static amrex::Real m_MG_tolerance_rel = 1.e-4;
    /** Absolute tolerance for the multigrid solver, when using the explicit solver */
    inline static amrex::Real m_MG_tolerance_abs = std::numeric_limits<amrex::Real>::min();
    /** Level of verbosity for the MG solver */
    inline static int m_MG_verbose = 0;
    /** Whether to use amrex MLMG solver */
    inline static bool m_use_amrex_mlmg = false;
    /** Whether the simulation uses a laser pulse */
    inline static bool m_use_laser = false;
    /** Background plasma density in SI, used to compute collisions, ionization,
     * or radiation reaction in normalized units */
    inline static amrex::Real m_background_density_SI = 0.;
    /** Time step for the beam evolution */
    amrex::Real m_dt = 0.0;
    /** Number of binary collisions */
    inline static int m_ncollisions = 0;
    /** Adaptive time step instance */
    AdaptiveTimeStep m_adaptive_time_step;
    /** Laser instance (soon to be multi laser container) */
    MultiLaser m_multi_laser;
    /** GridCurrent instance */
    GridCurrent m_grid_current;
#ifdef HIPACE_USE_OPENPMD
    /** openPMD writer instance */
    OpenPMDWriter m_openpmd_writer;
#endif
    /** the number of SALAME iterations to be done */
    int m_salame_n_iter = 3;
    /** if the SALAME-only field should be computed exactly with plasma particles */
    bool m_salame_do_advance = true;
    /** the slice index of the previous slice with SALAME */
    int m_salame_last_slice = -1;
    /** if the SALAME beam was overloaded in the last slice */
    bool m_salame_overloaded = false;
    /** the slice index of the previous slice with SALAME */
    amrex::Real m_salame_zeta_initial = 0;
    /** Parser for m_salame_target_func */
    amrex::Parser m_salame_parser;
    /** Function to get the target Ez field for SALAME */
    amrex::ParserExecutor<3> m_salame_target_func;

private:

#ifdef AMREX_USE_LINEAR_SOLVERS
    /** Linear operator for the explicit Bx and By solver */
    amrex::Vector<std::unique_ptr<amrex::MLALaplacian>> m_mlalaplacian;
    /** Geometric multigrid solver class, for the explicit Bx and By solver */
    amrex::Vector<std::unique_ptr<amrex::MLMG>> m_mlmg;
#endif
    /** hpmg solver for the explicit Bx and by solver */
    amrex::Vector<std::unique_ptr<hpmg::MultiGrid>> m_hpmg;
    /** Diagnostics */
    Diagnostic m_diags;
    /** User-input names of the binary collisions to be used */
    std::vector<std::string> m_collision_names;
    /** Vector of binary collisions */
    amrex::Vector< CoulombCollision > m_all_collisions;

    void InitDiagnostics (const int step);
    void FillFieldDiagnostics (const int lev, int islice);
    void FillBeamDiagnostics (const int step);
    void WriteDiagnostics (const int step);
    void FlushDiagnostics ();

    /** \brief get diagnostics Component names of Fields to output */
    amrex::Vector<std::string>& getDiagBeamNames () { return m_diags.getBeamNames(); }

    /** \brief Predictor-corrector loop to calculate Bx and By.
     * 1. an initial Bx and By value is guessed.
     * 2. Using this Bx and By values, the plasma particles are advanced to the next slice,
     *  and deposit their current there.
     * 3. With that current, Bx and By can be calculated.
     * 4. Mixing the calculated Bx and By with the previous guess a new Bx and By is calculated
     * 5. 2.-4. are repeated for a fixed number of iterations
     *
     * This modifies component Bx and By, of slice 1 in m_fields.m_slices
     * as well as the plasma particle force terms.
     *
     * \param[in] islice longitudinal slice
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] step current time step
     */
    void PredictorCorrectorLoopToSolveBxBy (const int islice, const int current_N_level,
                                            const int step);
};

#endif
