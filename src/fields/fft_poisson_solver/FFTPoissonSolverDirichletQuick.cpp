/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverDirichletQuick.H"
#include "fft/AnyFFT.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_BaseFabUtility.H>

FFTPoissonSolverDirichletQuick::FFTPoissonSolverDirichletQuick (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

template<class T> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real dst2_in (T&& in, int i, int j, int n) {
    return 2*i < n ? in(2*i, j) : -in(2*(n-i)-1, j);
}

template<class T> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real dst2_out (T&& in, int i, int j, int n, const amrex::GpuComplex<amrex::Real>* omega) {
    if (2*i+1 < n) {
        return - (in(i+1, j) * omega[i+1]).imag();
    } else {
        return (in(n-i-1, j) * omega[n-i-1]).real();
    }
}

template<class T> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::GpuComplex<amrex::Real> dst3_in (
    T&& in, int i, int j, int n, const amrex::GpuComplex<amrex::Real>* omega) {

    if (i == 0) {
        return {in(n-1, j), 0};
    } else {
        auto o = omega[i];
        o.m_imag = - o.m_imag;
        return o * amrex::GpuComplex<amrex::Real>{in(n-i-1, j), -in(i-1, j)};
    }
}

template<class T> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real dst3_out (T&& in, int i, int j, int n) {
    return i%2 == 0 ? in(i/2, j) : -in(n-1-i/2, j);
}

template<class T, class U> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void dst2_out_mult_dst3_in (
    T&& inout, U&& mult, int i, int j, int n, const amrex::GpuComplex<amrex::Real>* omega) {

    auto c = inout(i, j);

    if (i == 0) {
        c.m_real *= mult(n-1, j);
        c.m_imag = 0;
    } else {
        auto o = omega[i];
        c *= o;
        o.m_imag = - o.m_imag;
        c = o * amrex::GpuComplex<amrex::Real>{
            c.real() * mult(n-i-1, j),
            c.imag() * mult(i-1, j)
        };
    }

    inout(i, j) = c;
}

inline void
dst2_out_t_in (Array2<amrex::GpuComplex<amrex::Real>> const& in, Array2<amrex::Real> const& out,
               int nx, int ny, const amrex::GpuComplex<amrex::Real>* omega) {

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    // constexpr int tile_dim_x = 16;
    // constexpr int tile_dim_y = 32;
    // constexpr int block_rows_x = 8;
    // constexpr int block_rows_y = 16;

    // const int tile_begin_x = (nx-1)/2;

    // const int num_blocks_x = (nx - tile_begin_x + tile_dim_x - 1)/tile_dim_x;
    // const int num_blocks_y = (ny + tile_dim_y - 1)/tile_dim_y;

    // amrex::launch<tile_dim_x*block_rows_y>(num_blocks_x*num_blocks_y, amrex::Gpu::gpuStream(),
    //     [=] AMREX_GPU_DEVICE() noexcept
    //     {
    //         __shared__ amrex::GpuComplex<amrex::Real> tile_ptr[tile_dim_x * tile_dim_y];

    //         const int block_y = blockIdx.x / num_blocks_x;
    //         const int block_x = blockIdx.x - block_y*num_blocks_x;

    //         const int tile_start_x = tile_begin_x + block_x* tile_dim_x;
    //         const int tile_start_y = block_y * tile_dim_y;

    //         int thread_y = threadIdx.x / tile_dim_x;
    //         int thread_x = threadIdx.x - thread_y*tile_dim_x;

    //         #pragma unroll(2)
    //         for (; thread_y < tile_dim_y; thread_y += block_rows_y)
    //         {
    //             const int iout = tile_start_x + thread_x;
    //             const int jout = tile_start_y + thread_y;
    //             const int iin = nx-iout-1;
    //             const int jin = 2*jout < ny ? 2*jout : 2*(ny-jout)-1;

    //             if (iout < nx && jout < ny) {
    //                 auto val = in(iin, jin) * omega[iin];
    //                 if (2*jout < ny) {
    //                     val = -val;
    //                 }
    //                 tile_ptr[thread_x + thread_y * tile_dim_x] = val;
    //             }
    //         }

    //         __syncthreads();

    //         thread_x = threadIdx.x / tile_dim_y;
    //         thread_y = threadIdx.x - thread_x*tile_dim_y;

    //         #pragma unroll(2)
    //         for (; thread_x < tile_dim_x; thread_x += block_rows_x)
    //         {
    //             const int iout = tile_start_x + thread_x;
    //             const int jout = tile_start_y + thread_y;
    //             const int iin = nx-iout-1;
    //             const int iout2 = iin-1;
    //             const bool do_iout2 = iout2 >= 0 && iout2 != iout;

    //             if (iout < nx && jout < ny) {
    //                 auto val = tile_ptr[thread_x + thread_y * tile_dim_x];
    //                 out(jout, iout) = -val.real();
    //                 if (do_iout2) {
    //                     out(jout, iout2) = val.imag();
    //                 }
    //             }
    //         }
    //     });

    amrex::ParallelFor(amrex::BoxND<2>{{0, (nx-1)/2}, {ny-1, nx-1}},
        [=] AMREX_GPU_DEVICE (int j, int i){

            int iout = i;
            int jout = j;

            int iin = nx-iout-1;
            int iout2 = iin-1;
            bool do_iout2 = iout2 >= 0 && iout2 != iout;

            int jin = 2*jout < ny ? 2*jout : 2*(ny-jout)-1;

            auto val = in(iin, jin) * omega[iin];

            if (2*jout < ny) {
                val = -val;
            }

            out(jout, iout) = -val.real();
            if (do_iout2) {
                out(jout, iout2) = val.imag();
            }
        });

#else
    auto fuse_out_t = [=] AMREX_GPU_DEVICE (int i, int j) {
        return dst2_out(in, j, i, nx, omega);
    };

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {ny-1, nx-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            out(i, j) = dst2_in(fuse_out_t, i, j, ny);
        });
#endif
}

inline void
dst3_out_t_in (Array2<amrex::Real> const& in, Array2<amrex::GpuComplex<amrex::Real>> const& out,
               int nx, int ny, const amrex::GpuComplex<amrex::Real>* omega) {


    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx/2, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){

            int iout = i; // [0, nx/2]
            int jout = j; // [0, ny-1]

            int jin = jout%2 == 0 ? jout/2 : ny-1-jout/2;

            int iin1 = nx-iout-1;
            amrex::Real val1 = in(jin, iin1);
            amrex::Real val2 = 0;
            if (iout != 0) {
                int iin2 = iout - 1;
                val2 = in(jin, iin2);
            }

            if (jout%2 != 0) {
                val1 = - val1;
                val2 = - val2;
            }

            auto o = omega[iout];
            o.m_imag = - o.m_imag;
            out(iout, jout) = o * amrex::GpuComplex<amrex::Real>{val1, -val2};
        });


    // auto fuse_out_t = [=] AMREX_GPU_DEVICE (int i, int j) {
    //     return dst3_out(in, j, i, ny);
    // };

    // amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx/2, ny-1}},
    //     [=] AMREX_GPU_DEVICE (int i, int j){
    //         out(i, j) = dst3_in(fuse_out_t, i, j, nx, omega);
    //     });
}



void
FFTPoissonSolverDirichletQuick::define (amrex::BoxArray const& a_realspace_ba,
                                       amrex::DistributionMapping const& dm,
                                       amrex::Geometry const& gm )
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletQuick::define()");
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

    m_gm = gm;
    const amrex::Box fft_box = m_stagingArea[0].box();
    const amrex::IntVect fft_size = fft_box.length();
    const int nx = fft_size[0];
    const int ny = fft_size[1];
    const auto dx = gm.CellSizeArray();
    const amrex::Real dxsquared = dx[0]*dx[0];
    const amrex::Real dysquared = dx[1]*dx[1];
    const amrex::Real sine_x_factor = MathConst::pi / ( 2. * ( nx + 1 ));
    const amrex::Real sine_y_factor = MathConst::pi / ( 2. * ( ny + 1 ));

    const amrex::Real norm_fac = 1. / ((nx + 1.) * (ny + 1.));

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
    m_position_array.resize(nx*ny);
    m_fourier_array.resize(std::max((nx/2+1)*ny, (ny/2+1)*nx));

    // Allocate and initialize the FFT plans
    std::size_t fft1_area = m_x_r2cfft.Initialize(FFTType::R2C_1D_batched, nx, ny);
    std::size_t fft2_area = m_y_r2cfft.Initialize(FFTType::R2C_1D_batched, ny, nx);
    std::size_t fft3_area = m_x_c2rfft.Initialize(FFTType::C2R_1D_batched, nx, ny);
    std::size_t fft4_area = m_y_c2rfft.Initialize(FFTType::C2R_1D_batched, ny, nx);

    // Allocate work area for both FFTs
    m_fft_work_area.resize(std::max({fft1_area, fft2_area, fft3_area, fft4_area}));

    m_x_r2cfft.SetBuffers(m_position_array.dataPtr(), m_fourier_array.dataPtr(),
                          m_fft_work_area.dataPtr());
    m_y_r2cfft.SetBuffers(m_position_array.dataPtr(), m_fourier_array.dataPtr(),
                          m_fft_work_area.dataPtr());
    m_x_c2rfft.SetBuffers(m_fourier_array.dataPtr(), m_position_array.dataPtr(),
                          m_fft_work_area.dataPtr());
    m_y_c2rfft.SetBuffers(m_fourier_array.dataPtr(), m_position_array.dataPtr(),
                          m_fft_work_area.dataPtr());

    // set up prefactors for ToSine
    m_omega_x.resize(nx/2+1);
    amrex::GpuComplex<amrex::Real>* const omega_x_ptr = m_omega_x.dataPtr();
    amrex::ParallelFor(nx/2+1,
        [=] AMREX_GPU_DEVICE (int i) {
            auto [imag, real] = amrex::Math::sincospi(-i/amrex::Real(2*nx));
            omega_x_ptr[i] = {real, imag};
        });

    m_omega_y.resize(ny/2+1);
    amrex::GpuComplex<amrex::Real>* const omega_y_ptr = m_omega_y.dataPtr();
    amrex::ParallelFor(ny/2+1,
        [=] AMREX_GPU_DEVICE (int i) {
            auto [imag, real] = amrex::Math::sincospi(-i/amrex::Real(2*ny));
            omega_y_ptr[i] = {real, imag};
        });
}


void
FFTPoissonSolverDirichletQuick::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletQuick::SolvePoissonEquation()");

    const int nx = m_stagingArea[0].box().length(0); // initially contiguous
    const int ny = m_stagingArea[0].box().length(1); // contiguous after transpose

    Array2<amrex::Real> pos_arr {{m_stagingArea[0].dataPtr(), {0,0,0}, {nx,ny,1}, 1}};

    Array2<amrex::Real> real_arr {{m_position_array.dataPtr(), {0,0,0}, {nx,ny,1}, 1}};
    Array2<amrex::Real> real_arr_t {{m_position_array.dataPtr(), {0,0,0}, {ny,nx,1}, 1}};

    Array2<amrex::GpuComplex<amrex::Real>> comp_arr {{ m_fourier_array.dataPtr(), {0,0,0}, {nx/2+1,ny,1}, 1}};
    Array2<amrex::GpuComplex<amrex::Real>> comp_arr_t {{ m_fourier_array.dataPtr(), {0,0,0}, {ny/2+1,nx,1}, 1}};

    amrex::Box lhs_bx = lhs_mf[0].box();
    // shift box to handle ghost cells properly
    lhs_bx -= m_stagingArea[0].box().smallEnd();
    Array2<amrex::Real> lhs_arr {{lhs_mf[0].dataPtr(), amrex::begin(lhs_bx), amrex::end(lhs_bx), 1}};

    Array2<amrex::Real> mult_t {m_eigenvalue_matrix.array()};
    const amrex::GpuComplex<amrex::Real>* omega_x_ptr = m_omega_x.dataPtr();
    const amrex::GpuComplex<amrex::Real>* omega_y_ptr = m_omega_y.dataPtr();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            real_arr(i, j) = dst2_in(pos_arr, i, j, nx);
        });

    m_x_r2cfft.Execute();

    dst2_out_t_in(comp_arr, real_arr_t, nx, ny, omega_x_ptr);

    m_y_r2cfft.Execute();


    const amrex::Real sine_x_factor = 1. / ( 2. * ( nx + 1 ));
    const amrex::Real sine_y_factor = 1. / ( 2. * ( ny + 1 ));

    const amrex::Real norm_fac = -4. * (nx + 1.) * (ny + 1.);
    const amrex::Real invdxsq = m_gm.InvCellSize(0)*m_gm.InvCellSize(0)*norm_fac;
    const amrex::Real invdysq = m_gm.InvCellSize(1)*m_gm.InvCellSize(1)*norm_fac;

    auto mult_t_func = [=] AMREX_GPU_DEVICE (int i, int j) {
        amrex::Real x_fac = amrex::Math::sinpi(sine_x_factor * (j+1));
        amrex::Real y_fac = amrex::Math::sinpi(sine_y_factor * (i+1));

        amrex::Real k = x_fac*x_fac*invdxsq + y_fac*y_fac*invdysq;

        if (k != 0) {
            return 1 / k;
        } else {
            return amrex::Real(0);
        }
    };

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {ny/2, nx-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            dst2_out_mult_dst3_in(comp_arr_t, mult_t_func, i, j, ny, omega_y_ptr);
        });

    m_y_c2rfft.Execute();

    dst3_out_t_in(real_arr_t, comp_arr, nx, ny, omega_x_ptr);

    // auto real_arr_t_dst3_out_t = [=] AMREX_GPU_DEVICE (int i, int j) {
    //     return dst3_out(real_arr_t, j, i, ny);
    // };

    // amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx/2, ny-1}},
    //     [=] AMREX_GPU_DEVICE (int i, int j){
    //         comp_arr(i, j) = dst3_in(real_arr_t_dst3_out_t, i, j, nx, omega_x_ptr);
    //     });

    m_x_c2rfft.Execute();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            lhs_arr(i, j) = dst3_out(real_arr, i, j, nx);
        });
}
