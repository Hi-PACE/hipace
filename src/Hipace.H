#ifndef HIPACE_H_
#define HIPACE_H_

#include "fields/Fields.H"
#include "fields/fft_poisson_solver/FFTPoissonSolver.H"
#include "particles/MultiPlasma.H"
#include "particles/MultiBeam.H"
#include "particles/BeamParticleContainer.H"
#include "utils/AdaptiveTimeStep.H"
#include "utils/GridCurrent.H"
#include "utils/Constants.H"
#include "diagnostics/Diagnostic.H"

#include <AMReX_AmrCore.H>
#ifdef AMREX_USE_LINEAR_SOLVERS
#  include <AMReX_MLALaplacian.H>
#  include <AMReX_MLMG.H>
#endif

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#include "diagnostics/OpenPMDWriter.H"
#endif

#include <memory>

/** \brief Singleton class, that intialize, runs and finalizes the simulation */
class Hipace final : public amrex::AmrCore
{
public:
    /** Ctor: read general input parameters, call constructors of main member variables
     * and initialize longitudinal and transverse MPI communicators */
    Hipace ();

    /** Destructor */
    ~Hipace ();

    /** Get singleton instance */
    static Hipace& GetInstance ();

    /** Virtual functions need to be defined for pure virtual class AmrCore */
    void MakeNewLevelFromScratch (
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    /** Tag cells for refinement */
    virtual void ErrorEst (
                   int /*lev*/, amrex::TagBoxArray& /*tags*/,
                   amrex::Real /*time*/, int /*ngrow*/) final;

    /** Make a new level using provided BoxArray and DistributionMapping and
     * fill with interpolated coarse level data */
    void MakeNewLevelFromCoarse (
                                 int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                                 const amrex::DistributionMapping& /*dm*/) override {}

    /** Remake an existing level using provided BoxArray and DistributionMapping and fill
     * with existing fine and coarse data */
    void RemakeLevel (
                      int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                      const amrex::DistributionMapping& /*dm*/) override {}

    /** Delete level data */
    void ClearLevel (int /*lev*/) override {}

    /**\brief Apply some user-defined changes the to base grids.
     *
     * This function is only called by MakeNewGrids after computing a box array for
     * the coarsest level and before calling MakeNewLevelFromScratch.
     * For example, use this function if you want to remove covered grids on
     * the coarsest refinement level. */
    void PostProcessBaseGrids (amrex::BoxArray& ba0) const override;

    /** Init AmrCore and allocate beam and plasma containers */
    void InitData ();

    /** Run the simulation. This function contains the loop over time steps */
    void Evolve ();

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
     * \param[in] bins slice indexing
     * \param[in] only_ghost whether to send only ghost particles
     */
    void Notify (const int step, const int it, amrex::Vector<BeamBins>& bins, bool only_ghost=false);

    /** \brief When slices sent to rank downstream, free buffer memory and make buffer nullptr
     *
     * \param[in] it current box number
     * \param[in] only_ghost whether to pack only ghost particles (or only valid particles)
     */
    void NotifyFinish (const int it=0, bool only_ghost=false);

    /** \brief return whether rank is in the same transverse communicator w/ me
     *
     * \param[in] rank MPI rank to test
     */
    bool InSameTransverseCommunicator (int rank) const;

    /** \brief Dump simulation data to file
     *
     * \param[in] output_step current iteration
     * \param[in] it current box number
     * \param[in] call_type whether the beams or the fields should be written to file
     */
    void WriteDiagnostics (int output_step, const int it, const OpenPMDWriterCallType call_type);

    /** \brief Return a copy of member struct for physical constants */
    PhysConst get_phys_const () {return m_phys_const;};

    /** \brief Full evolve on 1 slice
     *
     * \param[in] islice slice number
     * \param[in] lev MR level
     * \param[in] ibox index of the current box to be calculated
     * \param[in] bins an amrex::DenseBins object that orders particles by slice
     */
    void SolveOneSlice (int islice, int lev, const int ibox,
                        amrex::Vector<BeamBins>& bins);

    /** \brief Reset plasma and field slice quantities to initial value.
     *
     * Typically done at the beginning of each iteration.
     *
     * \param[in] lev MR level
     */
    void ResetAllQuantities (int lev);

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

    /** All field data (3D array, slices) and field methods */
    Fields m_fields;
    /** Contains all beam species */
    MultiBeam m_multi_beam;
    /** Contains all plasma species */
    MultiPlasma m_multi_plasma;
    /** Number of time iterations */
    static int m_max_step;
    /** Time step for the beam evolution */
    static amrex::Real m_dt;
    /** Number of iterations between consecutive output dumps.
     * Default is -1, meaning no output */
    int m_output_period = -1;
    /** Physical time of the simulation. At the end of the time step, it is the physical time
     * at which the fields have been calculated. The beam is one step ahead. */
    static amrex::Real m_physical_time;
    /** Level of verbosity */
    static int m_verbose;
    /** Whether to use normalized units */
    static bool m_normalized_units;
    /** Struct containing physical constants (which values depends on the unit system, determined
     * at runtime): SI or normalized units. */
    PhysConst m_phys_const;
    /** Order of the field gather and current deposition shape factor in the transverse directions
     */
    static int m_depos_order_xy;
    /** Order of the field gather and current deposition shape factorin the longitudinal direction
     */
    static int m_depos_order_z;
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
    /** Whether to call amrex::Gpu::synchronize() around all profiler region */
    static bool m_do_device_synchronize;
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
    /** Adaptive time step instance */
    AdaptiveTimeStep m_adaptive_time_step;
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
    bool m_explicit = false;
    /**
     * \brief Solve for Bx an By in slice MF using the explicit solver
     *
     * The algorithm used was derived in [Wang, T. et al. arXiv preprint arXiv:2012.00881 (2020)],
     * it is implemented in the WAND-PIC quasistatic PIC code.
     *
     * \param[lev] MR level
     */
    void ExplicitSolveBxBy (const int lev);

    /** \brief Check that the ghost beam particles are in the proper box, and invalidate
     * those not in the right slice.
     *
     * \param[in] it Current box, the ghost slice of particles should be in the cell downstream.
     */
    void CheckGhostSlice (int it);

    amrex::RealVect patch_lo {0., 0., 0.}; /**< 3D array with lower ends of the refined grid */
    amrex::RealVect patch_hi {0., 0., 0.}; /**< 3D array with upper ends of the refined grid */
private:
    /** Pointer to current (and only) instance of class Hipace */
    static Hipace* m_instance;

    amrex::Geometry m_slice_geom;
    amrex::DistributionMapping m_slice_dm;
    amrex::BoxArray m_slice_ba;

#ifdef AMREX_USE_LINEAR_SOLVERS
    /** Linear operator for the explicit Bx and By solver */
    std::unique_ptr<amrex::MLALaplacian> m_mlalaplacian;
    /** Geometric multigrid solver class, for the explicit Bx and By solver */
    std::unique_ptr<amrex::MLMG> m_mlmg;
#endif
    /** Used to sort the beam particles into boxes for pipelining */
    amrex::Vector<BoxSorter> m_box_sorters;

    /** Diagnostics */
    Diagnostic m_diags;

    void ResizeFDiagFAB (const amrex::Box box, const int lev) { m_diags.ResizeFDiagFAB(box, lev); };
    void FillDiagnostics (int lev, int i_slice);
    /** \brief get diagnostics Component names of Fields to output */
    amrex::Vector<std::string>& getDiagComps () { return m_diags.getComps(); };
    /** \brief get diagnostics Component names of Fields to output */
    amrex::Vector<std::string>& getDiagBeamNames () { return m_diags.getBeamNames(); };
    /** \brief get diagnostics multifab */
    amrex::Vector<amrex::FArrayBox>& getDiagF () { return m_diags.getF(); };
    /** \brief get diagnostics geometry */
    amrex::Vector<amrex::Geometry>& getDiagGeom () { return m_diags.getGeom(); };
    /** \brief get slicing direction of the diagnostics */
    int getDiagSliceDir () { return m_diags.sliceDir(); };

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
     * \param[in] islice index of calculated slice
     * \param[in] lev current level
     * \param[in] bx current box to calculate in loop over longutidinal boxes
     * \param[in] bins Vector (over species) of particles sorted by slices
     * \param[in] ibox index of the current box
     */
    void PredictorCorrectorLoopToSolveBxBy (const int islice, const int lev, const amrex::Box bx,
                        amrex::Vector<BeamBins> bins,
                        const int ibox);

    void Ionisation (const int lev);

    /** \brief define Geometry, DistributionMapping and BoxArray for the slice.
     * The slice MultiFAB and the plasma particles are defined on this GDB.
     *
     * \param[in] ba BoxArray of the whole domain
     * \param[in] dm DistributionMappint of the whole domain
     */
    void DefineSliceGDB (const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

    int leftmostBoxWithParticles () const;
};

#endif
