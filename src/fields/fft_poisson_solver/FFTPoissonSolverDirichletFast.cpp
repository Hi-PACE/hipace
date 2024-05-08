/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverDirichletFast.H"
#include "fft/AnyFFT.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichletFast::FFTPoissonSolverDirichletFast (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

/** \brief Make Complex array out of Real array to prepare for fft.
 * out[idx] = -in[2*idx-2] + in[2*idx] + i*in[2*idx-1] for each column with
 * in[-1] = 0; in[-2] = -in[0]; in[n_data] = 0; in[n_data+1] = -in[n_data-1]
 *
 * \param[in] in input real array
 * \param[out] out output complex array
 * \param[in] n_data number of (contiguous) rows in position matrix
 * \param[in] n_batch number of (strided) columns in position matrix
 */
void ToComplex (const amrex::Real* const AMREX_RESTRICT in,
                amrex::GpuComplex<amrex::Real>* const AMREX_RESTRICT out,
                const int n_data, const int n_batch)
{
    const int n_half = (n_data+1)/2;
    if((n_data%2 == 1)) {
        amrex::ParallelFor({{0,0,0}, {n_half,n_batch-1,0}},
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                const int stride_in = n_data*j;
                const int i_is_zero = (i==0);
                const int i_is_n_half = (i==n_half);
                const amrex::Real real = -in[2*i-2+2*i_is_zero+stride_in]*(1-2*i_is_zero)
                                            +in[2*i-2*i_is_n_half+stride_in]*(1-2*i_is_n_half);
                const amrex::Real imag = in[2*i-1+i_is_zero-i_is_n_half+stride_in]
                                            *!i_is_zero*!i_is_n_half;
                out[i+(n_half+1)*j] = amrex::GpuComplex<amrex::Real>(real, imag);
            });
    } else {
        amrex::ParallelFor({{0,0,0}, {n_half,n_batch-1,0}},
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                const int stride_in = n_data*j;
                const int i_is_zero = (i==0);
                const int i_is_n_half = (i==n_half);
                const amrex::Real real = -in[2*i-2+2*i_is_zero+stride_in]*(1-2*i_is_zero)
                                            +in[2*i-i_is_n_half+stride_in]*!i_is_n_half;
                const amrex::Real imag = in[2*i-1+i_is_zero+stride_in]*!i_is_zero;
                out[i+(n_half+1)*j] = amrex::GpuComplex<amrex::Real>(real, imag);
            });
    }
}

/** \brief Make Sine-space Real array out of array from fft.
 * out[idx] = 0.5 *(in[n_data-idx] - in[idx+1] + (in[n_data-idx] + in[idx+1])/
 * (2*sin((idx+1)*pi/(n_data+1)))) for each column
 *
 * \param[in] in input real array
 * \param[out] out output real array
 * \param[in] n_data number of (contiguous) rows in position matrix
 * \param[in] n_batch number of (strided) columns in position matrix
 */
void ToSine (const amrex::Real* const AMREX_RESTRICT in, amrex::Real* const AMREX_RESTRICT out,
             const int n_data, const int n_batch)
{
    using namespace amrex::literals;

    const amrex::Real n_1_real_inv = 1._rt / (n_data+1._rt);
    const int n_1 = n_data+1;
    amrex::ParallelFor({{1,0,0}, {(n_data+1)/2,n_batch-1,0}},
        [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
        {
            const int i_rev = n_1-i;
            const int stride_in = n_1*j;
            const int stride_out = n_data*j;
            const amrex::Real in_a = in[i + stride_in];
            const amrex::Real in_b = in[i_rev + stride_in];
            out[i - 1 + stride_out] =
                0.5_rt*(in_b - in_a + (in_a + in_b) / (2._rt * amrex::Math::sinpi(i * n_1_real_inv)));
            out[i_rev - 1 + stride_out] =
                0.5_rt*(in_a - in_b + (in_a + in_b) / (2._rt * amrex::Math::sinpi(i_rev * n_1_real_inv)));
        });
}

/** \brief Transpose input matrix
 * out[idy][idx] = in[idx][idy]
 *
 * \param[in] in input real array
 * \param[out] out output real array
 * \param[in] n_data number of (contiguous) rows in input matrix
 * \param[in] n_batch number of (strided) columns in input matrix
 */
void Transpose (const amrex::Real* const AMREX_RESTRICT in,
                amrex::Real* const AMREX_RESTRICT out,
                const int n_data, const int n_batch)
{
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    constexpr int tile_dim = 32; //must be power of 2
    constexpr int block_rows = 8;
    const int num_blocks_x = (n_data + tile_dim - 1)/tile_dim;
    const int num_blocks_y = (n_batch + tile_dim - 1)/tile_dim;
    amrex::launch(num_blocks_x*num_blocks_y, tile_dim*block_rows,
        tile_dim*(tile_dim+1)*sizeof(amrex::Real), amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept
        {
            amrex::Gpu::SharedMemory<amrex::Real> gsm;
            amrex::Real* const tile = gsm.dataPtr();

            const int thread_x = threadIdx.x&(tile_dim-1);
            const int thread_y = threadIdx.x/tile_dim;
            const int block_y = blockIdx.x/num_blocks_x;
            const int block_x = blockIdx.x - block_y*num_blocks_x;
            int mat_x = block_x * tile_dim + thread_x;
            int mat_y = block_y * tile_dim + thread_y;

            for (int i = 0; i < tile_dim; i += block_rows) {
                if(mat_x < n_data && (mat_y+i) < n_batch) {
                    tile[(thread_y+i)*(tile_dim+1) + thread_x] = in[(mat_y+i)*n_data + mat_x];
                }
            }

            __syncthreads();

            mat_x = block_y * tile_dim + thread_x;
            mat_y = block_x * tile_dim + thread_y;

            for (int i = 0; i < tile_dim; i += block_rows) {
                if(mat_x < n_batch && (mat_y+i) < n_data) {
                    out[(mat_y+i)*n_batch + mat_x] = tile[thread_x*(tile_dim+1) + thread_y+i];
                }
            }
        });
#else
    amrex::ParallelFor({{0,0,0}, {n_batch-1, n_data-1, 0}},
        [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            out[i + n_batch*j] = in[j + n_data*i];
        });
#endif
}

void
FFTPoissonSolverDirichletFast::define (amrex::BoxArray const& a_realspace_ba,
                                       amrex::DistributionMapping const& dm,
                                       amrex::Geometry const& gm )
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletFast::define()");
    using namespace amrex::literals;

    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_realspace_ba.size() == 1, "Parallel FFT not supported yet");

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    // The stagingArea is also created from 0 to nx, because the real space array may have
    // an offset for levels > 0
    m_stagingArea = amrex::MultiFab(a_realspace_ba, dm, 1, Fields::m_poisson_nguards);
    m_tmpSpectralField = amrex::MultiFab(a_realspace_ba, dm, 1, Fields::m_poisson_nguards);
    m_eigenvalue_matrix = amrex::MultiFab(a_realspace_ba, dm, 1, Fields::m_poisson_nguards);
    m_stagingArea.setVal(0.0, Fields::m_poisson_nguards); // this is not required
    m_tmpSpectralField.setVal(0.0, Fields::m_poisson_nguards);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
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

    // Allocate 1d Array for 2d data or 2d transpose data
    const int real_1d_size = std::max((nx+1)*ny, (ny+1)*nx);
    const int complex_1d_size = std::max(((nx+1)/2+1)*ny, ((ny+1)/2+1)*nx);
    m_position_array.resize(real_1d_size);
    m_fourier_array.resize(complex_1d_size);

    // Allocate and initialize the FFT plans
    std::size_t fft_x_area = m_x_fft.Initialize(FFTType::C2R_1D_batched, nx+1, ny);
    std::size_t fft_y_area = m_y_fft.Initialize(FFTType::C2R_1D_batched, ny+1, nx);

    // Allocate work area for both FFTs
    m_fft_work_area.resize(std::max(fft_x_area, fft_y_area));

    m_x_fft.SetBuffers(m_fourier_array.dataPtr(), m_position_array.dataPtr(),
                       m_fft_work_area.dataPtr());
    m_y_fft.SetBuffers(m_fourier_array.dataPtr(), m_position_array.dataPtr(),
                       m_fft_work_area.dataPtr());
}


void
FFTPoissonSolverDirichletFast::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletFast::SolvePoissonEquation()");

    const int nx = m_stagingArea[0].box().length(0); // initially contiguous
    const int ny = m_stagingArea[0].box().length(1); // contiguous after transpose

    amrex::Real* const pos_arr = m_stagingArea[0].dataPtr();
    amrex::Real* const fourier_arr = m_tmpSpectralField[0].dataPtr();
    amrex::Real* const real_arr = m_position_array.dataPtr();
    amrex::GpuComplex<amrex::Real>* comp_arr = m_fourier_array.dataPtr();

    // 1D DST in x
    ToComplex(pos_arr, comp_arr, nx, ny);

    m_x_fft.Execute();

    ToSine(real_arr, pos_arr, nx, ny);

    Transpose(pos_arr, fourier_arr, nx, ny);

    // 1D DST in y
    ToComplex(fourier_arr, comp_arr, ny, nx);

    m_y_fft.Execute();

    ToSine(real_arr, pos_arr, ny, nx);

    Transpose(pos_arr, fourier_arr, ny, nx);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(m_stagingArea, DfltMfiTlng); mfi.isValid(); ++mfi ){
        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by eigenvalue_matrix
        Array2<amrex::Real> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        Array2<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);

        amrex::ParallelFor( mfi.growntilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                tmp_cmplx_arr(i,j) *= eigenvalue_matrix(i,j);
            });
    }

    // 1D DST in x
    ToComplex(fourier_arr, comp_arr, nx, ny);

    m_x_fft.Execute();

    ToSine(real_arr, fourier_arr, nx, ny);

    Transpose(fourier_arr, pos_arr, nx, ny);

    // 1D DST in y
    ToComplex(pos_arr, comp_arr, ny, nx);

    m_y_fft.Execute();

    ToSine(real_arr, fourier_arr, ny, nx);

    Transpose(fourier_arr, pos_arr, ny, nx);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(m_stagingArea, DfltMfiTlng); mfi.isValid(); ++mfi ){
        // Copy from the staging area to output array (and normalize)
        Array2<amrex::Real> tmp_real_arr = m_stagingArea.array(mfi);
        Array2<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lhs_mf.size() == 1,
                                         "Slice MFs must be defined on one box only");
        amrex::ParallelFor( lhs_mf[mfi].box() & mfi.growntilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                // Copy field
                lhs_arr(i,j) = tmp_real_arr(i,j);
            });
    }
}
