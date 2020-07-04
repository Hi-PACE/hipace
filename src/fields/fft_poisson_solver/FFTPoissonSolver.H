#ifndef FFT_POISSON_SOLVER_H_
#define FFT_POISSON_SOLVER_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include <AMReX_MultiFab.H>

#include <AMReX_GpuComplex.H>
// Declare type for fields in spectral fields
using SpectralField = amrex::FabArray< amrex::BaseFab <amrex::GpuComplex<amrex::Real>> >;

/**
 * \brief This class handles functions and data to perform transverse Fourier-based Poisson solves.
 *
 * For a given source S, it solves equation \Delta F = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
class FFTPoissonSolver
{
public:
    FFTPoissonSolver ( amrex::BoxArray const& realspace_ba,
                       amrex::DistributionMapping const& dm,
                       amrex::Geometry const& gm);

    FFTPoissonSolver () = default;

    void SolvePoissonEquation (amrex::MultiFab& lhs_mf);

    amrex::MultiFab& StagingArea() {return m_stagingArea;};
private:
    amrex::BoxArray m_spectralspace_ba;
    SpectralField m_tmpSpectralField; // contains (complex) field in Fourier space
    amrex::MultiFab m_stagingArea;    // contains (real) field in real space
    amrex::MultiFab m_inv_k2;         // 1/(kx^2 + ky^2)

    AnyFFT::FFTplans m_forward_plan, m_backward_plan;
};

#endif
