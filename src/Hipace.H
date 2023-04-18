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
    static int m_depos_order_xy;
    /** Order of the field gather and current deposition shape factor in the longitudinal direction
     */
    static int m_depos_order_z;
    /** Type of derivative used in explicit deposition. 0: analytic, 1: nodal, 2: centered
     */
    static int m_depos_derivative_type;
    /* if the loop over depos_order is included in the loop over particles
     */
    static bool m_outer_depos_loop;

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

    /** Destructor */
    ~Hipace ();

    /** Get singleton instance */
    static Hipace& GetInstance ();

    /** Init AmrCore and allocate beam and plasma containers */
    void InitData ();

    /** Run the simulation. This function contains the loop over time steps */
    void Evolve ();

    /** Make Geometry, DistributionMapping and BoxArray for all MR levels*/
    void MakeGeometry ();

    /** \brief Receive field slices from rank upstream
     *
     * Initialize a buffer (in pinned memory on Nvidia GPUs) for slices to be received (2 and 3),
     * MPI_Recv slices from the upstream rank into this buffer and copy from the buffer to
     * slice multifabs of current rank.
     *
     * \param[in] step current time step
     * \param[in] it index of the box for which data is received
     * \param[in] only_ghost whether to recv only ghost particles
     */
    void Wait (const int step, int it, bool only_ghost=false);

    /** \brief Send field slices to rank downstream
     *
     * Initialize a buffer (in pinned memory on Nvidia GPUs) for slices to be sent (2 and 3),
     * copy from slice multifabs of current rank to the buffer, MPI_Isend the buffer to
     * the rank downstream.
     *
     * \param[in] step current time step
     * \param[in] it current box number
     * \param[in] only_ghost whether to send only ghost particles
     */
    void Notify (const int step, const int it, bool only_ghost=false);

    /** \brief When slices sent to rank downstream, free buffer memory and make buffer nullptr
     *
     * \param[in] it current box number
     * \param[in] only_ghost whether to pack only ghost particles (or only valid particles)
     * \param[in] only_time whether only the time is send (and no beam particles)
     */
    void NotifyFinish (const int it=0, bool only_ghost=false, bool only_time=false);

    /** \brief Dump simulation data to file
     *
     * \param[in] output_step current iteration
     * \param[in] it current box number
     * \param[in] call_type whether the beams or the fields should be written to file
     */
    void WriteDiagnostics (int output_step, const int it, const OpenPMDWriterCallType call_type);

    /** \brief Return a copy of member struct for physical constants */
    PhysConst get_phys_const () {return m_phys_const;}

    /** \brief Full evolve on 1 slice
     *
     * \param[in] islice slice number
     * \param[in] islice_local local index of the slice
     * \param[in] step current time step
     */
    void SolveOneSlice (int islice, const int islice_local, int step);

    /** \brief Full evolve on 1 subslice with explicit solver
     *
     * The algorithm used was derived in [Wang, T. et al. arXiv preprint arXiv:2012.00881 (2020)],
     * it is implemented in the WAND-PIC quasistatic PIC code.
     *
     * \param[in] lev MR level
     * \param[in] step current time step
     * \param[in] islice longitudinal slice
     * \param[in] islice_local local index of the slice
     */
    void ExplicitSolveOneSubSlice (const int lev, const int step,
                                   const int islice, const int islice_local);

    /** \brief Full evolve on 1 subslice with predictor-corrector solver
     *
     * \param[in] lev MR level
     * \param[in] step current time step
     * \param[in] islice longitudinal slice
     * \param[in] islice_local local index of the slice
     */
    void PredictorCorrectorSolveOneSubSlice (const int lev, const int step,
                                             const int islice, const int islice_local);

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

    /** \brief reset all laser slices to 0 */
    void ResetLaser ();

    /**
       \brief Calls FillBoundary on charges and currents on WhichSlice::This
     *
     * \param[in] lev MR level
     */
    void FillBoundaryChargeCurrents (int lev);

    /**
       \brief Returns true on the head rank, otherwise false.
     */
    static bool HeadRank ()
    {
        return amrex::ParallelDescriptor::MyProc()==amrex::ParallelDescriptor::NProcs()-1;
    }

    /** Version of the HiPACE executable
     *
     * @return detailed version string
     */
    static std::string Version ();

    /** Transverse MPI communicator (for transverse exchanges in 1 slice in the presence of
     * transverse parallelization)
     */
    MPI_Comm m_comm_xy = MPI_COMM_NULL;
    /** Longitudinal MPI communicator (to send data downstream in the presence of
     * longitudinal parallelization)
     */
    MPI_Comm m_comm_z = MPI_COMM_NULL;
    /** Number of processors in the transverse x direction */
    int m_numprocs_x = 1;
    /** Number of processors in the transverse y direction */
    int m_numprocs_y = 1;
    /** Number of processors in the longitudinal z direction */
    int m_numprocs_z = 0;
    /** My rank in the transverse communicator */
    int m_rank_xy = 0;
    /** My rank in the longitudinal communicator */
    int m_rank_z = 0;
    /** Max number of grid size in the longitudinal direction */
    int m_boxes_in_z = 1;
    /** Send buffer for particle longitudinal parallelization (pipeline) */
    char* m_psend_buffer = nullptr;
    char* m_psend_buffer_ghost = nullptr;
    amrex::Real m_tsend_buffer {-1.};
    /** Send buffer for the number of particles for each beam (pipeline) */
    amrex::Vector<int> m_np_snd = amrex::Vector<int>(0);
    amrex::Vector<int> m_np_snd_ghost = amrex::Vector<int>(0);
    /** status of the number of particles send request */
    MPI_Request m_nsend_request = MPI_REQUEST_NULL;
    MPI_Request m_nsend_request_ghost = MPI_REQUEST_NULL;
    /** status of the particle send request */
    MPI_Request m_psend_request = MPI_REQUEST_NULL;
    MPI_Request m_psend_request_ghost = MPI_REQUEST_NULL;
    /** status of the physical time send request */
    MPI_Request m_tsend_request = MPI_REQUEST_NULL;
    // Laser buffer
    amrex::Real* m_lsend_buffer = nullptr;
    // status of laser send
    MPI_Request m_lsend_request = MPI_REQUEST_NULL;
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
    static Hipace* m_instance;
    /** Whether to use normalized units */
    static bool m_normalized_units;
    /** All field data (3D array, slices) and field methods */
    Fields m_fields;
    /** Contains all beam species */
    MultiBeam m_multi_beam;
    /** Contains all plasma species */
    MultiPlasma m_multi_plasma;
    /** Number of time iterations */
    static int m_max_step;
    /** Maximum simulation time */
    static amrex::Real m_max_time;
    /** Time step for the beam evolution */
    static amrex::Real m_dt;
    /** Physical time of the simulation. At the end of the time step, it is the physical time
     * at which the fields have been calculated. The beam is one step ahead. */
    static amrex::Real m_physical_time;
    /** Physical time at the beginning of the simulation */
    static amrex::Real m_initial_time;
    /** Level of verbosity */
    static int m_verbose;
    /** Relative transverse B field error tolerance in the predictor corrector loop
     */
    static amrex::Real m_predcorr_B_error_tolerance;
    /** Maximum number of iterations in the predictor corrector loop
     */
    static int m_predcorr_max_iterations;
    /** Average number of iterations in the predictor corrector loop
     */
    amrex::Real m_predcorr_avg_iterations = 0.;
    /** Average transverse B field error in the predictor corrector loop
     */
    amrex::Real m_predcorr_avg_B_error = 0.;
    /** Mixing factor between the transverse B field iterations in the predictor corrector loop
     */
    static amrex::Real m_predcorr_B_mixing_factor;
    /** Whether the beams deposit Jx and Jy */
    static bool m_do_beam_jx_jy_deposition;
    /** Whether the jz-c*rho contribution of the beam is computed and used. If not, jz-c*rho=0 is assumed */
    static bool m_do_beam_jz_minus_rho;
    /** Whether to use tiling for particle operations */
    static bool m_do_tiling;
    /** How much the box is coarsened for beam injection, to avoid exceeding max int in cell count.
     * Otherwise, changing this parameter only will not affect the simulation results. */
    static int m_beam_injection_cr;
    /** Slope of external focusing fields applied to beam particles.
     * The fields applied are ExmBy = m_external_ExmBy_slope*x, and same in y. */
    static amrex::Real m_external_ExmBy_slope;
    /** Slope of external accelerating fields applied to beam particles.
     * The fields applied are Ez = m_external_Ez_slope*z. */
    static amrex::Real m_external_Ez_slope;
    /** Uniform accelerating fields applied to beam particles. */
    static amrex::Real m_external_Ez_uniform;
    /** Relative tolerance for the multigrid solver, when using the explicit solver */
    static amrex::Real m_MG_tolerance_rel;
    /** Absolute tolerance for the multigrid solver, when using the explicit solver */
    static amrex::Real m_MG_tolerance_abs;
    /** Level of verbosity for the MG solver */
    static int m_MG_verbose;
    /** Whether to use amrex MLMG solver */
    static bool m_use_amrex_mlmg;
    /** Whether the simulation uses a laser pulse */
    static bool m_use_laser;
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
    /** index of the most downstream box to send that contains beam particles.
     * Used to avoid send/recv for empty data */
    int m_leftmost_box_snd = std::numeric_limits<int>::max();
    /** index of the most downstream box to be received that contains beam particles.
     * Used to avoid send/recv for empty data */
    int m_leftmost_box_rcv = std::numeric_limits<int>::max();
    /** Whether to skip communications of boxes that contain no beam particles */
    int m_skip_empty_comms = false;
    /** Whether the explicit field solver is used */
    bool m_explicit = true;
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

    /** \brief Check that the ghost beam particles are in the proper box, and invalidate
     * those not in the right slice.
     *
     * \param[in] it Current box, the ghost slice of particles should be in the cell downstream.
     */
    void CheckGhostSlice (int it);

private:

#ifdef AMREX_USE_LINEAR_SOLVERS
    /** Linear operator for the explicit Bx and By solver */
    amrex::Vector<std::unique_ptr<amrex::MLALaplacian>> m_mlalaplacian;
    /** Geometric multigrid solver class, for the explicit Bx and By solver */
    amrex::Vector<std::unique_ptr<amrex::MLMG>> m_mlmg;
#endif
    /** hpmg solver for the explicit Bx and by solver */
    amrex::Vector<std::unique_ptr<hpmg::MultiGrid>> m_hpmg;
private:
    /** Diagnostics */
    Diagnostic m_diags;

    /** \brief resizes the diagnostic fab to the correct box in a loop over boxes
     *
     * \param[in] it index of box to be resized to
     * \param[in] step current time step
     */
    void ResizeFDiagFAB (const int it, const int step);
    void FillDiagnostics (const int lev, int i_slice);
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
     * \param[in] islice_local local index of the slice
     * \param[in] lev current level
     * \param[in] step current time step
     */
    void PredictorCorrectorLoopToSolveBxBy (const int islice, const int islice_local,
                                            const int lev, const int step);

    int leftmostBoxWithParticles () const;
};

#endif
