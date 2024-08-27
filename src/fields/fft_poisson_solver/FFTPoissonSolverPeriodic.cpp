/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Remi Lehe
 * WeiqunZhang
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverPeriodic.H"
#include "fft/AnyFFT.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverPeriodic::FFTPoissonSolverPeriodic (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

void
FFTPoissonSolverPeriodic::define ( amrex::BoxArray const& realspace_ba,
                                   amrex::DistributionMapping const& dm,
                                   amrex::Geometry const& gm )
{
    HIPACE_PROFILE("FFTPoissonSolverPeriodic::define()");
    using namespace amrex::literals;

    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(realspace_ba.size() == 1, "Parallel FFT not supported yet");

    // Create the box array that corresponds to spectral space
    amrex::BoxList spectral_bl; // Create empty box list
    // Loop over boxes and fill the box list
    for (int i=0; i < realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        amrex::Box realspace_bx = realspace_ba[i];
        realspace_bx.grow(Fields::m_poisson_nguards);
        amrex::IntVect fft_size = realspace_bx.length();
        // Because the spectral solver uses real-to-complex FFTs, we only
        // need the positive k values along the fastest axis
        // (first axis for AMReX Fortran-order arrays) in spectral space.
        // This effectively reduces the size of the spectral space by half
        // see e.g. the FFTW documentation for real-to-complex FFTs
        amrex::IntVect spectral_bx_size = fft_size;
        spectral_bx_size[0] = fft_size[0]/2 + 1;
        // Define the corresponding box
        amrex::Box spectral_bx = amrex::Box( amrex::IntVect::TheZeroVector(),
                          spectral_bx_size - amrex::IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
    }
    amrex::BoxArray spectralspace_ba{};
    spectralspace_ba.define( std::move(spectral_bl) );

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    m_stagingArea = amrex::MultiFab(realspace_ba, dm, 1, Fields::m_poisson_nguards);
    m_tmpSpectralField = SpectralField(spectralspace_ba, dm, 1, 0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    // Calculate the array of inv_k2
    amrex::Real dkx = 2*MathConst::pi/gm.ProbLength(0);
    amrex::Real dky = 2*MathConst::pi/gm.ProbLength(1);
    m_inv_k2 = amrex::MultiFab(spectralspace_ba, dm, 1, 0);
    // Loop over boxes and calculate inv_k2 in each box
    for (amrex::MFIter mfi(m_inv_k2, DfltMfi); mfi.isValid(); ++mfi ){
        Array2<amrex::Real> inv_k2_arr = m_inv_k2.array(mfi);
        amrex::Box const& bx = mfi.validbox();  // The lower corner of the "2D" slice Box is zero.
        int const Ny = bx.length(1);
        int const mid_point_y = (Ny+1)/2;
        amrex::ParallelFor(to2D(bx), [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            // kx is always positive (first axis of the real-to-complex FFT)
            amrex::Real kx = dkx*i;
            // The first half of ky is positive ; the other is negative
            amrex::Real ky = (j<mid_point_y) ? dky*j : dky*(j-Ny);
            if ((i!=0) && (j!=0)) {
                inv_k2_arr(i,j) = 1._rt/(kx*kx + ky*ky);
            } else {
                // Avoid division by 0
                inv_k2_arr(i,j) = 0._rt;
            }
        });
    }

    // Allocate and initialize the FFT plans
    amrex::IntVect fft_size = m_stagingArea[0].box().length();
    std::size_t fwd_area = m_forward_fft.Initialize(FFTType::R2C_2D, fft_size[0], fft_size[1]);
    std::size_t bkw_area = m_backward_fft.Initialize(FFTType::C2R_2D, fft_size[0], fft_size[1]);

    // Allocate work area for both FFTs
    m_fft_work_area.resize(std::max(fwd_area, bkw_area));

    m_forward_fft.SetBuffers(m_stagingArea[0].dataPtr(), m_tmpSpectralField[0].dataPtr(),
                             m_fft_work_area.dataPtr());
    m_backward_fft.SetBuffers(m_tmpSpectralField[0].dataPtr(), m_stagingArea[0].dataPtr(),
                              m_fft_work_area.dataPtr());
}


void
FFTPoissonSolverPeriodic::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverPeriodic::SolvePoissonEquation()");

    m_forward_fft.Execute();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(m_tmpSpectralField, DfltMfiTlng); mfi.isValid(); ++mfi ){
        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by inv_k2
        Array2<amrex::GpuComplex<amrex::Real>> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        Array2<amrex::Real> inv_k2_arr = m_inv_k2.array(mfi);
        amrex::ParallelFor( to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept {
                tmp_cmplx_arr(i,j) *= -inv_k2_arr(i,j);
            });
    }

    m_backward_fft.Execute();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(m_stagingArea, DfltMfiTlng); mfi.isValid(); ++mfi ){
        // Copy from the staging area to output array (and normalize)
        Array2<amrex::Real> tmp_real_arr = m_stagingArea.array(mfi);
        Array2<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        const amrex::Box fft_box = m_stagingArea[mfi].box();
        const amrex::Real inv_N = 1./fft_box.numPts();
        amrex::ParallelFor( to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept {
                // Copy and normalize field
                lhs_arr(i,j) = inv_N*tmp_real_arr(i,j);
            });

    }
}
