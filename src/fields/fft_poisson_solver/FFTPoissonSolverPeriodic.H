#ifndef FFT_POISSON_SOLVER_PERIODIC_H_
#define FFT_POISSON_SOLVER_PERIODIC_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "FFTPoissonSolver.H"

#include <AMReX_MultiFab.H>
#include <AMReX_GpuComplex.H>

/** Declare type for fields in spectral fields */
using SpectralField = amrex::FabArray< amrex::BaseFab <amrex::GpuComplex<amrex::Real>> >;

/**
 * \brief This class handles functions and data to perform transverse Fourier-based Poisson solves.
 *
 * For a given source S, it solves equation Laplacian(F) = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S directly in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
class FFTPoissonSolverPeriodic : public FFTPoissonSolver
{
public:
    FFTPoissonSolverPeriodic ( amrex::BoxArray const& realspace_ba,
                               amrex::DistributionMapping const& dm,
                               amrex::Geometry const& gm);// :
//        FFTPoissonSolver(realspace_ba, dm, gm) {};

    // FFTPoissonSolver () = default;

    /**
     * \brief Define real space and spectral space boxes and multifabs, multiplier
     * coefficients inv_k2 to solve Poisson equation and FFT plans.
     * Currently only works with a single box, i.e., serial FFT.
     *
     * \param[in] realspace_ba BoxArray on which the FFT is executed.
     * \param[in] dm DistributionMapping for the BoxArray.
     * \param[in] gm Geometry, contains the box dimensions.
     */
    virtual void define ( amrex::BoxArray const& realspace_ba,
                          amrex::DistributionMapping const& dm,
                          amrex::Geometry const& gm) override final;

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    virtual void SolvePoissonEquation (amrex::MultiFab& lhs_mf) override final;

private:
    /** Spectral fields, contains (complex) field in Fourier space */
    SpectralField m_tmpSpectralField;
    /** Multifab containing 1/(kx^2 + ky^2), to solve Poisson equation. */
    amrex::MultiFab m_inv_k2;
    /** FFT plans */
    AnyFFT::FFTplans m_forward_plan, m_backward_plan;
};

#endif
