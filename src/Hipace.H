#ifndef HIPACE_H_
#define HIPACE_H_

#include "fields/Fields.H"
#include "fields/fft_poisson_solver/FFTPoissonSolver.H"
#include "particles/PlasmaParticleContainer.H"
#include "particles/BeamParticleContainer.H"
#include "Constants.H"

#include <AMReX_AmrCore.H>

/** \brief Singleton class, that intialize, runs and finalizes the simulation */
class Hipace final : public amrex::AmrCore
{
public:
    /** Ctor: read general input parameters, call constructors of main member variables
     * and initialize longitudinal and transverse MPI communicators */
    Hipace ();

    ~Hipace ();

    static Hipace& GetInstance ();

    /** Virtual functions need to be defined for pure virtual class AmrCore */
    void MakeNewLevelFromScratch (
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    void ErrorEst (
                   int /*lev*/, amrex::TagBoxArray& /*tags*/, amrex::Real /*time*/, int /*ngrow*/) override {}

    void MakeNewLevelFromCoarse (
                                 int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                                 const amrex::DistributionMapping& /*dm*/) override {}

    void RemakeLevel (
                      int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                      const amrex::DistributionMapping& /*dm*/) override {}

    void ClearLevel (int /*lev*/) override {}

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
    /** When slices sent to rank downstream, free buffer memory and make buffer nullptr */
    void NotifyFinish ();

    /** return whether rank is in the same transverse communicator w/ me
     *
     * \param[in] rank MPI rank to test
     */
    bool InSameTransverseCommunicator (int rank) const;

    /** \brief Dump simulation data to file
     *
     * \param[in] step current iteration
     */
    void WriteDiagnostics (int step);

    /** Return a copy of member struct for physical constants */
    PhysConst get_phys_const () {return m_phys_const;};

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
    /** status of the send request */
    MPI_Request m_send_request = MPI_REQUEST_NULL;

    /** All field data (3D array, slices) and field methods */
    Fields m_fields;
    /** Class to handle transverse FFT Poisson solver on 1 slice */
    FFTPoissonSolver m_poisson_solver;
    /** Particle container for a beam */
    BeamParticleContainer m_beam_container;
    /** Particle container for the plasma */
    PlasmaParticleContainer m_plasma_container;
    /** Number of time iterations */
    int m_max_step = 0;
    /** Whether to dump output data or not */
    bool m_do_plot = true;

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

private:
    /** Pointer to current (and only) instance of class Hipace */
    static Hipace* m_instance;

    /** \brief Compute Bx on the slice container from J by solving a Poisson equation
     *
     * \param[in] lev current level
     */
    void SolvePoissonBx (const int lev);
    /** \brief Compute By on the slice container from J by solving a Poisson equation
     *
     * \param[in] lev current level
     */
    void SolvePoissonBy (const int lev);
};

/** \brief Type of current deposition, either in the current slice or in the next one
 * in the predictor-corrector loop
 */
enum struct CurrentDepoType { DepositThisSlice, DepositNextSlice };

#endif
