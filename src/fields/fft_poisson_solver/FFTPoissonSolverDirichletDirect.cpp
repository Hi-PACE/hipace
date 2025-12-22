/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverDirichletDirect.H"
#include "fft/AnyFFT.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichletDirect::FFTPoissonSolverDirichletDirect (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm,
    bool is_even)
{
    define(realspace_ba, dm, gm, is_even);
}

void
FFTPoissonSolverDirichletDirect::define (amrex::BoxArray const& a_realspace_ba,
                                         amrex::DistributionMapping const& dm,
                                         amrex::Geometry const& gm,
                                         bool is_even)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletDirect::define()");
    using namespace amrex::literals;

    m_is_even = is_even;

    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_realspace_ba.size() == 1, "Parallel FFT not supported yet");

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    // The stagingArea is also created from 0 to nx, because the real space array may have
    // an offset for levels > 0
    m_stagingArea = amrex::MultiFab(a_realspace_ba, dm, 1, 0);
    m_tmpSpectralField = amrex::MultiFab(a_realspace_ba, dm, 1, 0);
    m_eigenvalue_matrix = amrex::MultiFab(a_realspace_ba, dm, 1, 0);
    m_stagingArea.setVal(0.0); // this is not required
    m_tmpSpectralField.setVal(0.0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    const amrex::Box fft_box = m_stagingArea[0].box();
    const amrex::IntVect fft_size = fft_box.length();
    const int nx = fft_size[0];
    const int ny = fft_size[1];
    const int logical_nx = is_even ? nx : nx + 1;
    const int logical_ny = is_even ? ny : ny + 1;

    const amrex::Real sine_x_factor = 1._rt / ( 2._rt * logical_nx);
    const amrex::Real sine_y_factor = 1._rt / ( 2._rt * logical_ny);
    const amrex::Real norm_fac = -16._rt * logical_nx * logical_ny;
    const amrex::Real invdxsq = gm.InvCellSize(0)*gm.InvCellSize(0)*norm_fac;
    const amrex::Real invdysq = gm.InvCellSize(1)*gm.InvCellSize(1)*norm_fac;

    // Calculate the array of m_eigenvalue_matrix
    for (amrex::MFIter mfi(m_eigenvalue_matrix, DfltMfi); mfi.isValid(); ++mfi ){
        Array2<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);
        amrex::IntVect lo = fft_box.smallEnd();
        amrex::ParallelFor(to2D(fft_box),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                /* fast poisson solver diagonal x and y coeffs */
                const amrex::Real x_fac = amrex::Math::sinpi(sine_x_factor * (i - lo[0] + 1));
                const amrex::Real y_fac = amrex::Math::sinpi(sine_y_factor * (j - lo[1] + 1));
                const amrex::Real k = x_fac * x_fac * invdxsq + y_fac * y_fac * invdysq;

                if (k != 0._rt) {
                    eigenvalue_matrix(i,j) = 1._rt / k;
                } else {
                    // Avoid division by 0
                    eigenvalue_matrix(i,j) = 0._rt;
                }
            });
    }

    // Allocate and initialize the FFT plans
    std::size_t fwd_area = m_forward_fft.Initialize(
        is_even ? FFTType::R2R_2D_DST2 : FFTType::R2R_2D_DST1, fft_size[0], fft_size[1]);
    std::size_t bkw_area = m_backward_fft.Initialize(
        is_even ? FFTType::R2R_2D_DST3 : FFTType::R2R_2D_DST1, fft_size[0], fft_size[1]);

    // Allocate work area for both FFTs
    m_fft_work_area.resize(std::max(fwd_area, bkw_area));

    m_forward_fft.SetBuffers(m_stagingArea[0].dataPtr(), m_tmpSpectralField[0].dataPtr(),
                             m_fft_work_area.dataPtr());
    m_backward_fft.SetBuffers(m_tmpSpectralField[0].dataPtr(), m_stagingArea[0].dataPtr(),
                              m_fft_work_area.dataPtr());
}


void
FFTPoissonSolverDirichletDirect::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletDirect::SolvePoissonEquation()");

    m_forward_fft.Execute();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(m_stagingArea, DfltMfiTlng); mfi.isValid(); ++mfi ){
        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by eigenvalue_matrix
        Array2<amrex::Real> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        Array2<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);

        amrex::ParallelFor( to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept {
                tmp_cmplx_arr(i,j) *= eigenvalue_matrix(i,j);
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
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lhs_mf.size() == 1,
                                         "Slice MFs must be defined on one box only");
        amrex::ParallelFor( to2D(lhs_mf[mfi].box() & mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept {
                // Copy field
                lhs_arr(i,j) = tmp_real_arr(i,j);
            });
    }
}
