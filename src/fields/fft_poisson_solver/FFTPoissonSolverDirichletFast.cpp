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
 * \param[in] i x index to compute
 * \param[in] j y index to compute
 * \param[in] n_half highest index of the output array, equal to (n_data+1)/2
 * \param[in] n_data number of (contiguous) rows in position matrix
 */
template<class T> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::GpuComplex<amrex::Real> to_complex (T&& in, int i, int j, int n_half, int n_data) {
    amrex::Real real = 0;
    amrex::Real imag = 0;
    if (i == 0) {
        real = 2*in(2*i, j);
    } else if (i == n_half) {
        if (n_data & 1) { // n_data is odd
            real = -2*in(2*i-2, j);
        } else {
            real = -in(2*i-2, j);
            imag = in(2*i-1, j);
        }
    } else {
        real = in(2*i, j) - in(2*i-2, j);
        imag = in(2*i-1, j);
    }
    return {real, imag};
}

/** \brief Make Sine-space Real array out of array from fft.
 * out[idx] = 0.5 *(in[n_data-idx] - in[idx+1] + (in[n_data-idx] + in[idx+1])/
 * (2*sin((idx+1)*pi/(n_data+1)))) for each column
 *
 * \param[in] in input real array
 * \param[in] i x index to compute
 * \param[in] j y index to compute
 * \param[in] n_data number of (contiguous) rows in position matrix
 * \param[in] sine_factor prefactor for ToSine equal to 1/(2*sin((idx+1)*pi/(n_data+1)))
 */
template<class T> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real to_sine (T&& in, int i, int j, int n_data, const amrex::Real* sine_factor) {
    const amrex::Real in_a = in(i+1, j);
    const amrex::Real in_b = in(n_data-i, j);
    // possible optimization:
    // iterate over the elements is such a way that each thread computes (i,j) and (n_data-i-1,j)
    // so in_a and in_b can be reused
    return amrex::Real(0.5)*(in_b - in_a + (in_a + in_b) * sine_factor[i]);
}

void ToComplex (const Array2<amrex::Real> in, const Array2<amrex::GpuComplex<amrex::Real>> out,
                const int n_data, const int n_batch)
{
    const int n_half = (n_data+1)/2;
    amrex::ParallelFor(amrex::BoxND<2>{{0,0}, {n_half,n_batch-1}},
        [=] AMREX_GPU_DEVICE(int i, int j) noexcept
        {
            out(i, j) = to_complex(in, i, j, n_half, n_data);
        });
}

void ToSine (const Array2<amrex::Real> in, const Array2<amrex::Real> out,
             const amrex::Real* sine_factor, const int n_data, const int n_batch)
{
    amrex::ParallelFor(amrex::BoxND<2>{{0,0}, {n_data-1,n_batch-1}},
        [=] AMREX_GPU_DEVICE(int i, int j) noexcept
        {
            out(i, j) = to_sine(in, i, j, n_data, sine_factor);
        });
}

void ToSine_Transpose_ToComplex (const Array2<amrex::Real> in,
                                 const Array2<amrex::GpuComplex<amrex::Real>> out,
                                 const amrex::Real* sine_factor, const int n_data, const int n_batch)
{
    const int n_half = (n_batch+1)/2;

    auto transpose_to_sine = [=] AMREX_GPU_DEVICE (int i, int j) {
        return to_sine(in, j, i, n_data, sine_factor);
    };

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    constexpr int tile_dim_x = 16;
    constexpr int tile_dim_x_ex = 34;
    constexpr int tile_dim_y = 32;
    constexpr int block_rows_x = 8;
    constexpr int block_rows_y = 16;

    const int nx = n_half + 1;
    const int ny = n_data;

    const int nx_sin = n_data;
    const int ny_sin = n_batch;

    const int num_blocks_x = (nx + tile_dim_x - 1)/tile_dim_x;
    const int num_blocks_y = (ny + tile_dim_y - 1)/tile_dim_y;

    amrex::launch<tile_dim_x*block_rows_y>(num_blocks_x*num_blocks_y, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept
        {
            __shared__ amrex::Real tile_ptr[tile_dim_x_ex * tile_dim_y];

            const int block_y = blockIdx.x / num_blocks_x;
            const int block_x = blockIdx.x - block_y*num_blocks_x;

            const int tile_begin_x = 2 * block_x * tile_dim_x - 2;
            const int tile_begin_y = block_y * tile_dim_y;

            const int tile_end_x = tile_begin_x + tile_dim_x_ex;
            const int tile_end_y = tile_begin_y + tile_dim_y;

            Array2<amrex::Real> shared{{tile_ptr, {tile_begin_x, tile_begin_y, 0},
                                                  {tile_end_x, tile_end_y, 1}, 1}};

            {
                const int thread_x = threadIdx.x / tile_dim_y;
                const int thread_y = threadIdx.x - thread_x*tile_dim_y;

                for (int tx = thread_x; tx < tile_dim_x_ex; tx += block_rows_x) {
                    const int i = tile_begin_x + tx;
                    const int j = tile_begin_y + thread_y;

                    if (j < nx_sin && i < ny_sin && i >= 0 ) {
                        shared(i, j) = transpose_to_sine(i, j);
                    }
                }
            }

            __syncthreads();

            {
                const int thread_y = threadIdx.x / tile_dim_x;
                const int thread_x = threadIdx.x - thread_y*tile_dim_x;

                for (int ty = thread_y; ty < tile_dim_y; ty += block_rows_y) {
                    const int i = block_x * tile_dim_x + thread_x;
                    const int j = tile_begin_y + ty;

                    if (i < nx && j < ny) {
                        out(i, j) = to_complex(shared, i, j, n_half, n_batch);
                    }
                }
            }
        });
#else
    amrex::ParallelFor(amrex::BoxND<2>{{0,0}, {n_half,n_data-1}},
        [=] AMREX_GPU_DEVICE(int i, int j) noexcept
        {
            out(i, j) = to_complex(transpose_to_sine, i, j, n_half, n_batch);
        });
#endif
}

void ToSine_Mult_ToComplex (const Array2<amrex::Real> in,
                            const Array2<amrex::GpuComplex<amrex::Real>> out,
                            const Array2<amrex::Real> eigenvalue,
                            const amrex::Real* sine_factor, const int n_data, const int n_batch)
{
    const int n_half = (n_data+1)/2;

    auto mult_to_sine = [=] AMREX_GPU_DEVICE (int i, int j) {
        return eigenvalue(i, j) * to_sine(in, i, j, n_data, sine_factor);
    };

    amrex::ParallelFor(amrex::BoxND<2>{{0,0}, {n_half,n_batch-1}},
        [=] AMREX_GPU_DEVICE(int i, int j) noexcept
        {
            out(i, j) = to_complex(mult_to_sine, i, j, n_half, n_data);
        });
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
    m_stagingArea = amrex::MultiFab(a_realspace_ba, dm, 1, 0);
    m_stagingArea.setVal(0.0); // this is not required

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
    m_eigenvalue_matrix.resize({{0,0,0}, {ny-1,nx-1,0}});
    Array2<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array();
    amrex::ParallelFor(amrex::BoxND<2>{{0,0}, {ny-1,nx-1}},
        [=] AMREX_GPU_DEVICE (int j, int i) noexcept
        {
            /* fast poisson solver diagonal x coeffs */
            amrex::Real sinex_sq = std::sin(( i + 1 ) * sine_x_factor) * std::sin(( i + 1 ) * sine_x_factor);
            /* fast poisson solver diagonal y coeffs */
            amrex::Real siney_sq = std::sin(( j + 1 ) * sine_y_factor) * std::sin(( j + 1 ) * sine_y_factor);

            if ((sinex_sq!=0) && (siney_sq!=0)) {
                eigenvalue_matrix(j,i) = norm_fac / ( -4.0_rt * ( sinex_sq / dxsquared + siney_sq / dysquared ));
            } else {
                // Avoid division by 0
                eigenvalue_matrix(j,i) = 0._rt;
            }
        });

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

    // set up prefactors for ToSine
    m_sine_x_factor.resize(nx);
    amrex::Real* const sine_x_ptr = m_sine_x_factor.dataPtr();
    amrex::ParallelFor(nx,
        [=] AMREX_GPU_DEVICE (int i) {
            sine_x_ptr[i] = 1._rt / (2._rt * amrex::Math::sinpi((i + 1._rt) / (nx + 1._rt)));
        });

    m_sine_y_factor.resize(ny);
    amrex::Real* const sine_y_ptr = m_sine_y_factor.dataPtr();
    amrex::ParallelFor(ny,
        [=] AMREX_GPU_DEVICE (int i) {
            sine_y_ptr[i] = 1._rt / (2._rt * amrex::Math::sinpi((i + 1._rt) / (ny + 1._rt)));
        });
}


void
FFTPoissonSolverDirichletFast::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletFast::SolvePoissonEquation()");

    const int nx = m_stagingArea[0].box().length(0); // initially contiguous
    const int ny = m_stagingArea[0].box().length(1); // contiguous after transpose

    Array2<amrex::Real> pos_arr {{m_stagingArea[0].dataPtr(), {0,0,0}, {nx,ny,1}, 1}};

    Array2<amrex::Real> real_arr {{m_position_array.dataPtr(), {0,0,0}, {nx+1,ny,1}, 1}};
    Array2<amrex::Real> real_arr_t {{m_position_array.dataPtr(), {0,0,0}, {ny+1,nx,1}, 1}};

    Array2<amrex::GpuComplex<amrex::Real>> comp_arr {{ m_fourier_array.dataPtr(), {0,0,0}, {(nx+1)/2+1,ny,1}, 1}};
    Array2<amrex::GpuComplex<amrex::Real>> comp_arr_t {{ m_fourier_array.dataPtr(), {0,0,0}, {(ny+1)/2+1,nx,1}, 1}};

    // 1D DST in x
    ToComplex(pos_arr, comp_arr, nx, ny);

    m_x_fft.Execute();

    // 1D DST in y
    ToSine_Transpose_ToComplex(real_arr, comp_arr_t, m_sine_x_factor.dataPtr(), nx, ny);

    m_y_fft.Execute();

    // 1D DST in y
    ToSine_Mult_ToComplex(real_arr_t, comp_arr_t, m_eigenvalue_matrix.array(),
                          m_sine_y_factor.dataPtr(), ny, nx);

    m_y_fft.Execute();

    // 1D DST in x
    ToSine_Transpose_ToComplex(real_arr_t, comp_arr, m_sine_y_factor.dataPtr(), ny, nx);

    m_x_fft.Execute();

    amrex::Box lhs_bx = lhs_mf[0].box();
    // shift box to handle ghost cells properly
    lhs_bx -= m_stagingArea[0].box().smallEnd();
    Array2<amrex::Real> lhs_arr {{lhs_mf[0].dataPtr(), amrex::begin(lhs_bx), amrex::end(lhs_bx), 1}};

    ToSine(real_arr, lhs_arr, m_sine_x_factor.dataPtr(), nx, ny);
}
