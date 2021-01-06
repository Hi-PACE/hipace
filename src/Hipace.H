#ifndef HIPACE_H_
#define HIPACE_H_

#include "fields/Fields.H"
#include "fields/fft_poisson_solver/FFTPoissonSolver.H"
#include "particles/PlasmaParticleContainer.H"
#include "particles/MultiBeam.H"
#include "particles/BeamParticleContainer.H"
#include "utils/AdaptiveTimeStep.H"
#include "utils/Constants.H"

#include <AMReX_AmrCore.H>

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#include "diagnostics/OpenPMDWriter.H"
namespace io = openPMD;
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
    void ErrorEst (
                   int /*lev*/, amrex::TagBoxArray& /*tags*/,
                   amrex::Real /*time*/, int /*ngrow*/) override {}

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
     */
    void Wait ();
    /** \brief Send field slices to rank downstream
     *
     * Initialize a buffer (in pinned memory on Nvidia GPUs) for slices to be sent (2 and 3),
     * copy from slice multifabs of current rank to the buffer, MPI_Isend the buffer to
     * the rank downstream.
     */
    void Notify ();
    /** \brief When slices sent to rank downstream, free buffer memory and make buffer nullptr */
    void NotifyFinish ();

    /** \brief return whether rank is in the same transverse communicator w/ me
     *
     * \param[in] rank MPI rank to test
     */
    bool InSameTransverseCommunicator (int rank) const;

    /** \brief Dump simulation data to file
     *
     * \param[in] output_step current iteration
     * \param[in] force_output if true, dump output regardless of output_period
     */
    void WriteDiagnostics (int output_step, bool force_output=false);

    /** \brief Return a copy of member struct for physical constants */
    PhysConst get_phys_const () {return m_phys_const;};

    /** \brief Full evolve on 1 slice
     *
     * \param[in] islice slice number
     * \param[in] lev MR level
     * \param[in] bins collections of beam particle indices sorted per slice
     */
    void SolveOneSlice (
        int islice, int lev,
        amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins);

    /** \brief Reset plasma and field slice quantities to initial value.
     *
     * Typically done at the beginning of each iteration.
     *
     * \param[in] lev MR level
     */
    void ResetAllQuantities (int lev);

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
    int m_grid_size_z = 0;
    /** Send buffer for longitudinal parallelization (pipeline) */
    amrex::Real* m_send_buffer = nullptr;
    /** Send buffer for particle longitudinal parallelization (pipeline) */
    char* m_psend_buffer = nullptr;
    /** status of the send request */
    MPI_Request m_send_request = MPI_REQUEST_NULL;
    /** status of the particle send request */
    MPI_Request m_psend_request = MPI_REQUEST_NULL;

    /** All field data (3D array, slices) and field methods */
    Fields m_fields;
    /** Particle container for a beam */
    MultiBeam m_multi_beam;
    /** Particle container for the plasma */
    PlasmaParticleContainer m_plasma_container;
    /** Number of time iterations */
    int m_max_step = 0;
    /** Time step for the beam evolution */
    static amrex::Real m_dt;
    /** Number of iterations between consecutive output dumps.
     * Default is -1, meaning no output */
    int m_output_period = -1;
    /** Last iteration that was written to file.
     * This is stored to make sure we don't write the last iteration multiple times. */
    int m_last_output_dumped = -1;
    /** Physical time of the simulation. At the end of the time step, it is the physical time
     * at which the fields have been calculated. The beam is one step ahead. */
    amrex::Real m_physical_time = 0.;
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
    /** Mixing factor between the transverse B field iterations in the predictor corrector loop
     */
    static amrex::Real m_predcorr_B_mixing_factor;
    /** whether to deposit the beam density per slice, into the slice multifab.
     * If not, when running on GPU, the 3D arrays must be on the GPU. */
    static bool m_slice_beam;
    /** whether the 3D array stays in host memory (Pinned memory) or is allowed
     * to be in device memory (allocated in Managed memory).
     * If true, required m_slice_beam = true. */
    static bool m_3d_on_host;
    /** Whether to call amrex::Gpu::synchronize() around all profiler region */
    static bool m_do_device_synchronize;
    /** whether to output the central y=0 slice (if true), or the whole 3D array (if false) */
    static bool m_slice_F_xz;
    /** Whether to dump plasma particles */
    static bool m_output_plasma;
    /** How much the box is coarsened for beam injection, to avoid exceeding max int in cell count.
     * Otherwise, changing this parameter only will not affect the simulation results. */
    static int m_beam_injection_cr;
    /** Strength of external focusing fields applied to beam particles.
     * The fields applied are ExmBy = m_external_focusing_field_strength*x, and same in y. */
    static amrex::Real m_external_focusing_field_strength;
    /** Strength of external accelerating fields applied to beam particles.
     * The fields applied are Ez = m_external_accel_field_strength*z. */
    static amrex::Real m_external_accel_field_strength;
    /** Adaptive time step instance */
    AdaptiveTimeStep m_adaptive_time_step;
#ifdef HIPACE_USE_OPENPMD
    /** openPMD writer instance */
    OpenPMDWriter m_openpmd_writer;
#endif

private:
    /** Pointer to current (and only) instance of class Hipace */
    static Hipace* m_instance;

    amrex::Geometry m_slice_geom;
    amrex::DistributionMapping m_slice_dm;
    amrex::BoxArray m_slice_ba;

    /** \brief Predictor-corrector loop to calculate Bx and By.
     * 1. an initial Bx and By value is guessed.
     * 2. Using this Bx and By values, the plasma particles are advanced to the next slice,
     *  and deposit their current there.
     * 3. With that current, Bx and By can be calculated.
     * 4. Mixing the calculated Bx and By with the previous guess a new Bx and By is calculated
     * 5. 2.-4. are repeated for a fixed number of iterations
     *
     * This modifies component FieldComps::Bx and By, of slice 1 in m_fields.m_slices
     * as well as the plasma particle force terms.
     *
     * \param[in] islice index of calculated slice
     * \param[in] lev current level
     */
    void PredictorCorrectorLoopToSolveBxBy (const int islice, const int lev);

    /** \brief define Geometry, DistributionMapping and BoxArray for the slice.
     * The slice MultiFAB and the plasma particles are defined on this GDB.
     *
     * \param[in] ba BoxArray of the whole domain
     * \param[in] dm DistributionMappint of the whole domain
     */
    void DefineSliceGDB (const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
};

#endif
