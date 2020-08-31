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

    /** \brief Compute ExmBy and EypBx on the slice container from J by solving a Poisson equation
     * ExmBy and EypBx are solved in the same function because both rely on Psi.
     *
     * \param[in] lev current level
     */
    void SolvePoissonExmByAndEypBx (const int lev);
    /** \brief Compute Ez on the slice container from J by solving a Poisson equation
     *
     * \param[in] lev current level
     */
    void SolvePoissonEz (const int lev);
    /** \brief Compute Bx on the slice container from J by solving a Poisson equation
     *
     * \param[in,out] Bx_iter Bx field during current iteration of the predictor-corrector loop
     * \param[in] lev current level
     */
    void SolvePoissonBx (amrex::MultiFab& Bx_iter, const int lev);
    /** \brief Compute By on the slice container from J by solving a Poisson equation
     *
     * \param[in,out] By_iter By field during current iteration of the predictor-corrector loop
     * \param[in] lev current level
     */
    void SolvePoissonBy (amrex::MultiFab& By_iter, const int lev);
    /** \brief Compute Bz on the slice container from J by solving a Poisson equation
     *
     * \param[in] lev current level
     */
    void SolvePoissonBz (const int lev);

    /** \brief Sets the initial guess of the B field from the two previous slices
     *
     * This modifies component FieldComps::Bx or By of slice 1 in m_fields.m_slices
     *
     * \param[in] lev current level
     */
    void InitialBfieldGuess (const int lev);

    /** \brief Mixes the B field with the calculated current and previous iteration
     * of it and shifts the current to the previous iteration afterwards.
     * This modifies component FieldComps::Bx or By of slice 1 in m_fields.m_slices
     *
     * \param[in] B_iter B field during current iteration of the predictor-corrector loop
     * \param[in,out] B_prev_iter B field during previous iteration of the pred.-cor. loop
     * \param[in] field_comp field component to be mixed (usually FieldComps::Bx or FieldComps::By)
     * \param[in] lev current level
     */
     void MixAndShiftBfields (const amrex::MultiFab& B_iter, amrex::MultiFab& B_prev_iter,
                              const int field_comp, const int lev);

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
     * \param[in] lev current level
     */
    void PredictorCorrectorLoopToSolveBxBy (const int lev);
};

/** \brief Type of current deposition, either in the current slice or in the next one
 * in the predictor-corrector loop
 */
enum struct ToSlice { This, Next };

#endif
