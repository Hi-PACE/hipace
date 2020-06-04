#ifndef FFT_POISSON_SOLVER_H_
#define FFT_POISSON_SOLVER_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include <AMReX_MultiFab.H>

#include <AMReX_GpuComplex.H>
// Declare type for fields in spectral fields
using SpectralField = amrex::FabArray< amrex::BaseFab <amrex::GpuComplex<amrex::Real>> >;

class FFTPoissonSolver
{
public:
    FFTPoissonSolver ( amrex::BoxArray const& realspace_ba,
                       amrex::DistributionMapping const& dm,
                       amrex::Geometry const& gm);

    FFTPoissonSolver () = default;

    void SolvePoissonEquation ( amrex::MultiFab const& rhs_mf,
                                amrex::MultiFab& lhs_mf );
private:
    amrex::BoxArray m_spectralspace_ba;
    SpectralField m_tmpSpectralField; // contains (complex) field in Fourier space
    amrex::MultiFab m_tmpRealField;   // contains (real) field in real space
    amrex::MultiFab m_inv_k2;         // 1/(kx^2 + ky^2)

    AnyFFT::FFTplans m_forward_plan, m_backward_plan;
};

#endif
