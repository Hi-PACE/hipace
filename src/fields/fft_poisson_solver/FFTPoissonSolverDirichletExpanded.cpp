/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverDirichletExpanded.H"
#include "fft/AnyFFT.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichletExpanded::FFTPoissonSolverDirichletExpanded (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

void ExpandR2R (amrex::FArrayBox& dst, const amrex::FArrayBox& src)
{
    // This function expands
    //
    //  1  2  3
    //  4  5  6
    //  7  8  9
    //
    // into
    //
    //  0  0  0  0  0  0  0  0
    //  0  1  2  3  0 -3 -2 -1
    //  0  4  5  6  0 -6 -5 -4
    //  0  7  8  9  0 -9 -8 -7
    //  0  0  0  0  0  0  0  0
    //  0 -7 -8 -9  0  9  8  7
    //  0 -4 -5 -6  0  6  5  4
    //  0 -1 -2 -3  0  3  2  1
    amrex::Box bx = src.box();
    bx.growLo(0, 1);
    bx.growLo(1, 1);
    const int lox = bx.smallEnd(0);
    const int loy = bx.smallEnd(1);
    const int nx = bx.length(0);
    const int ny = bx.length(1);
    const int refx = dst.box().bigEnd(0)+lox+1;
    const int refy = dst.box().bigEnd(1)+loy+1;
    const Array2<amrex::Real const> src_array = src.array();
    const Array2<amrex::Real> dst_array = dst.array();

    amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int)
        {
            if (i == lox || j == loy) {
                dst_array(i, j) = 0;
                dst_array(i, j+ny) = 0;
                dst_array(i+nx, j) = 0;
                dst_array(i+nx, j+ny) = 0;
            } else {
                const amrex::Real val = src_array(i, j);
                /* upper left quadrant */
                dst_array(i, j) = val;
                /* lower left quadrant */
                dst_array(i, refy-j) = -val;
                /* upper right quadrant */
                dst_array(refx-i, j) = -val;
                /* lower right quadrant */
                dst_array(refx-i, refy-j) = val;
            }
        });
}

void Shrink_Mult_Expand (amrex::FArrayBox& dst,
                         const amrex::BaseFab<amrex::GpuComplex<amrex::Real>>& src,
                         const amrex::FArrayBox& eigenvalue)
{
    // This function combines ShrinkC2R -> multiply with eigenvalue -> ExpandR2R
    amrex::Box bx = eigenvalue.box();
    bx.growLo(0, 1);
    bx.growLo(1, 1);
    const int lox = bx.smallEnd(0);
    const int loy = bx.smallEnd(1);
    const int nx = bx.length(0);
    const int ny = bx.length(1);
    const int refx = dst.box().bigEnd(0)+lox+1;
    const int refy = dst.box().bigEnd(1)+loy+1;
    const Array2<amrex::GpuComplex<amrex::Real> const> src_array = src.array();
    const Array2<amrex::Real> dst_array = dst.array();
    const Array2<amrex::Real const> eigenvalue_array= eigenvalue.array();

    amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int)
        {
            if (i == lox || j == loy) {
                dst_array(i, j) = 0;
                dst_array(i, j+ny) = 0;
                dst_array(i+nx, j) = 0;
                dst_array(i+nx, j+ny) = 0;
            } else {
                const amrex::Real val = -src_array(i, j).real() * eigenvalue_array(i, j);
                /* upper left quadrant */
                dst_array(i, j) = val;
                /* lower left quadrant */
                dst_array(i, refy-j) = -val;
                /* upper right quadrant */
                dst_array(refx-i, j) = -val;
                /* lower right quadrant */
                dst_array(refx-i, refy-j) = val;
            }
        });
}

void ShrinkC2R (amrex::FArrayBox& dst, const amrex::BaseFab<amrex::GpuComplex<amrex::Real>>& src,
                amrex::Box bx)
{
    const Array2<amrex::GpuComplex<amrex::Real> const> src_array = src.array();
    const Array2<amrex::Real> dst_array = dst.array();
    amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int)
        {
            /* upper left quadrant */
            dst_array(i,j) = -src_array(i, j).real();
        });
}

void
FFTPoissonSolverDirichletExpanded::define (amrex::BoxArray const& a_realspace_ba,
                                           amrex::DistributionMapping const& dm,
                                           amrex::Geometry const& gm )
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletExpanded::define()");
    using namespace amrex::literals;

    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_realspace_ba.size() == 1, "Parallel FFT not supported yet");

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    // The stagingArea is also created from 0 to nx, because the real space array may have
    // an offset for levels > 0
    m_stagingArea = amrex::MultiFab(a_realspace_ba, dm, 1, Fields::m_poisson_nguards);
    m_eigenvalue_matrix = amrex::MultiFab(a_realspace_ba, dm, 1, Fields::m_poisson_nguards);
    m_stagingArea.setVal(0.0, Fields::m_poisson_nguards); // this is not required

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");

    const amrex::Box fft_box = m_stagingArea[0].box();
    const amrex::IntVect fft_size = fft_box.length();
    const int nx = fft_size[0];
    const int ny = fft_size[1];
    const auto dx = gm.CellSizeArray();
    const amrex::Real dxsquared = dx[0]*dx[0];
    const amrex::Real dysquared = dx[1]*dx[1];
    const amrex::Real sine_x_factor = MathConst::pi / ( 2. * ( nx + 1 ));
    const amrex::Real sine_y_factor = MathConst::pi / ( 2. * ( ny + 1 ));

    // Normalization of FFTW's 'DST-I' discrete sine transform (FFTW_RODFT00)
    // This normalization is used regardless of the sine transform library
    const amrex::Real norm_fac = 0.5 / ( 2 * (( nx + 1 ) * ( ny + 1 )));

    // Calculate the array of m_eigenvalue_matrix
    for (amrex::MFIter mfi(m_eigenvalue_matrix, DfltMfi); mfi.isValid(); ++mfi ){
        Array2<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);
        amrex::IntVect lo = fft_box.smallEnd();
        amrex::ParallelFor(
            fft_box, [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
                {
                    /* fast poisson solver diagonal x coeffs */
                    amrex::Real sinex_sq = std::sin(( i - lo[0] + 1 ) * sine_x_factor) * std::sin(( i - lo[0] + 1 ) * sine_x_factor);
                    /* fast poisson solver diagonal y coeffs */
                    amrex::Real siney_sq = std::sin(( j - lo[1] + 1 ) * sine_y_factor) * std::sin(( j - lo[1] + 1 ) * sine_y_factor);

                    if ((sinex_sq!=0) && (siney_sq!=0)) {
                        eigenvalue_matrix(i,j) = norm_fac / ( -4.0 * ( sinex_sq / dxsquared + siney_sq / dysquared ));
                    } else {
                        // Avoid division by 0
                        eigenvalue_matrix(i,j) = 0._rt;
                    }
                });
    }

    // Allocate expanded_position_array Real of size (2*nx+2, 2*ny+2)
    // Allocate expanded_fourier_array Complex of size (nx+2, 2*ny+2)
    amrex::Box expanded_position_box {{-1, -1, 0}, {2*nx, 2*ny, 0}};
    amrex::Box expanded_fourier_box {{-1, -1, 0}, {nx, 2*ny, 0}};
    // shift box to match rest of fields
    expanded_position_box += fft_box.smallEnd();
    expanded_fourier_box += fft_box.smallEnd();

    m_expanded_position_array.resize(expanded_position_box);
    m_expanded_fourier_array.resize(expanded_fourier_box);

    m_expanded_position_array.setVal<amrex::RunOn::Device>(0._rt);

    // Allocate and initialize the FFT plan
    std::size_t wrok_size = m_fft.Initialize(FFTType::R2C_2D, expanded_position_box.length(0),
                                             expanded_position_box.length(1));

    // Allocate work area for the FFT
    m_fft_work_area.resize(wrok_size);

    m_fft.SetBuffers(m_expanded_position_array.dataPtr(), m_expanded_fourier_array.dataPtr(),
                     m_fft_work_area.dataPtr());
}


void
FFTPoissonSolverDirichletExpanded::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletExpanded::SolvePoissonEquation()");
    using namespace amrex::literals;

    ExpandR2R(m_expanded_position_array, m_stagingArea[0]);

    m_fft.Execute();

    Shrink_Mult_Expand(m_expanded_position_array, m_expanded_fourier_array, m_eigenvalue_matrix[0]);

    m_fft.Execute();

    ShrinkC2R(lhs_mf[0], m_expanded_fourier_array, m_stagingArea[0].box());
}
