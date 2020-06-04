#ifndef FFT_POISSON_SOLVER_H_
#define FFT_POISSON_SOLVER_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include <AMReX_MultiFab.H>

// Declare type for fields in spectral fields
using SpectralField = amrex::FabArray< amrex::BaseFab <AnyFFT::Complex> >;

class FFTPoissonSolver
{
public:
    FFTPoissonSolver();

    void SolvePoissonEquation ( amrex::MultiFab const input_mf,
                                amrex::MultiFab output_mf );
private:

    SpectralField m_tmpSpectralField; // contains (complex) field in Fourier space
    amrex::MultiFab m_tmpRealField;   // contains (real) field in real space
    amrex::MultiFab m_inv_k2;         // 1/(kx^2 + ky^2)

    AnyFFT::FFTplans m_forward_plan, m_backward_plan;
};

#endif
