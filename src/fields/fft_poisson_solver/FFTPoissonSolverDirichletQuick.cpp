/* Copyright 2025
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

namespace {

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

inline void
dst2_out_mult_dst3_in (Array2<amrex::GpuComplex<amrex::Real>> const& inout, int nx, int ny,
                       const amrex::GpuComplex<amrex::Real>* omega,
                       const amrex::Real * eig_x, const amrex::Real * eig_y) {
    auto mult = [=] AMREX_GPU_DEVICE (int i, int j) {
        const amrex::Real k = eig_x[i] + eig_y[j];
        if (k != 0) {
            return 1 / k;
        } else {
            return amrex::Real(0);
        }
    };

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {ny/2, nx-1}},
        [=] AMREX_GPU_DEVICE (int j, int i){
            auto c = inout(j, i);

            if (j == 0) {
                c.m_real *= mult(i, ny-1);
                c.m_imag = 0;
            } else {
                auto o = omega[j];
                auto m1 = mult(i, ny-j-1);
                auto m2 = mult(i, j-1);
                c *= o;
                o.m_imag = - o.m_imag;
                c = o * amrex::GpuComplex<amrex::Real>{
                    c.real() * m1,
                    c.imag() * m2
                };
            }

            inout(j, i) = c;
        });
}

inline void
dst2_out_t_in (Array2<amrex::GpuComplex<amrex::Real>> const& in, Array2<amrex::Real> const& out,
               int nx, int ny, const amrex::GpuComplex<amrex::Real>* omega) {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    static constexpr int tile_dim_x = 16;
    static constexpr int tile_dim_y = 64;
    static constexpr int elem_per_thread = 2;
    static constexpr int block_rows_x = tile_dim_x / elem_per_thread;
    static constexpr int block_rows_y = tile_dim_y / elem_per_thread;

    const int tile_begin_x = (nx-1)/2;

    const int num_blocks_x = (nx - tile_begin_x + tile_dim_x - 1)/tile_dim_x;
    const int num_blocks_y = (ny + tile_dim_y - 1)/tile_dim_y;

    amrex::launch<tile_dim_x*block_rows_y>(num_blocks_x*num_blocks_y, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept
        {
            __shared__ amrex::Real tile_real[tile_dim_x][tile_dim_y+1];
            __shared__ amrex::Real tile_imag[tile_dim_x][tile_dim_y+1];

            const int block_x = blockIdx.x / num_blocks_y;
            const int block_y = blockIdx.x - block_x*num_blocks_y;

            const int tile_start_x = tile_begin_x + block_x* tile_dim_x;
            const int tile_start_y = block_y * tile_dim_y;

            int thread_y = threadIdx.x / tile_dim_x;
            int thread_x = threadIdx.x - thread_y*tile_dim_x;

            #pragma unroll elem_per_thread
            for (; thread_y < tile_dim_y; thread_y += block_rows_y)
            {
                const int thread_xr = tile_dim_x - thread_x - 1;
                const int iout = tile_start_x + thread_xr;
                const int jout = tile_start_y + thread_y;
                const int iin = nx-iout-1;
                const int jin = 2*jout < ny ? 2*jout : 2*(ny-jout)-1;

                if (iout < nx && jout < ny) {
                    auto val = in(iin, jin) * omega[iin];
                    if (2*jout < ny) {
                        val = -val;
                    }
                    tile_real[thread_xr][thread_y] = val.real();
                    tile_imag[thread_xr][thread_y] = val.imag();
                }
            }

            __syncthreads();

            thread_x = threadIdx.x / tile_dim_y;
            thread_y = threadIdx.x - thread_x*tile_dim_y;

            #pragma unroll elem_per_thread
            for (; thread_x < tile_dim_x; thread_x += block_rows_x)
            {
                const int iout = tile_start_x + thread_x;
                const int jout = tile_start_y + thread_y;
                const int iin = nx-iout-1;
                const int iout2 = iin-1;
                const bool do_iout2 = iout2 >= 0 && iout2 != iout;

                if (iout < nx && jout < ny) {
                    out(jout, iout) = -tile_real[thread_x][thread_y];
                    if (do_iout2) {
                        out(jout, iout2) = tile_imag[thread_x][thread_y];
                    }
                }
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
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    static constexpr int tile_dim_x = 16;
    static constexpr int tile_dim_y = 64;
    static constexpr int elem_per_thread = 2;
    static constexpr int block_rows_x = tile_dim_x / elem_per_thread;
    static constexpr int block_rows_y = tile_dim_y / elem_per_thread;

    const int tile_end_x = nx/2 + 1;

    const int num_blocks_x = (tile_end_x + tile_dim_x - 1)/tile_dim_x;
    const int num_blocks_y = (ny + tile_dim_y - 1)/tile_dim_y;

    amrex::launch<tile_dim_x*block_rows_y>(num_blocks_x*num_blocks_y, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept
        {
            __shared__ amrex::Real tile_real[tile_dim_y][tile_dim_x+1];
            __shared__ amrex::Real tile_imag[tile_dim_y][tile_dim_x+1];

            const int block_y = blockIdx.x / num_blocks_x;
            const int block_x = blockIdx.x - block_y*num_blocks_x;

            const int tile_start_x = block_x * tile_dim_x;
            const int tile_start_y = block_y * tile_dim_y;

            int thread_x = threadIdx.x / tile_dim_y;
            int thread_y = threadIdx.x - thread_x*tile_dim_y;

            #pragma unroll elem_per_thread
            for (; thread_x < tile_dim_x; thread_x += block_rows_x)
            {
                const int thread_yp = (thread_y*2) % tile_dim_y + (thread_y*2) / tile_dim_y;
                const int iout = tile_start_x + thread_x;
                const int jout = tile_start_y + thread_yp;
                const int jin = jout%2 == 0 ? jout/2 : ny-1-jout/2;
                const int iin1 = nx-iout-1;
                const int iin2 = iout - 1;

                if (iout < tile_end_x && jout < ny) {
                    amrex::GpuComplex<amrex::Real> val {
                        in(jin, iin1),
                        iout != 0 ? -in(jin, iin2) : 0
                    };
                    auto o = omega[iout];
                    o.m_imag = - o.m_imag;
                    if (jout%2 != 0) {
                        val = -val;
                    }
                    val *= o;
                    tile_real[thread_yp][thread_x] = val.real();
                    tile_imag[thread_yp][thread_x] = val.imag();
                }
            }

            __syncthreads();

            thread_y = threadIdx.x / tile_dim_x;
            thread_x = threadIdx.x - thread_y*tile_dim_x;

            #pragma unroll elem_per_thread
            for (; thread_y < tile_dim_y; thread_y += block_rows_y)
            {
                const int iout = tile_start_x + thread_x;
                const int jout = tile_start_y + thread_y;

                if (iout < tile_end_x && jout < ny) {
                    out(iout, jout) = {
                        tile_real[thread_y][thread_x],
                        tile_imag[thread_y][thread_x]
                    };
                }
            }
        });

#else
    auto fuse_out_t = [=] AMREX_GPU_DEVICE (int i, int j) {
        return dst3_out(in, j, i, ny);
    };

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx/2, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            out(i, j) = dst3_in(fuse_out_t, i, j, nx, omega);
        });
#endif
}

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

    const amrex::Box fft_box = m_stagingArea[0].box();
    const amrex::IntVect fft_size = fft_box.length();
    const int nx = fft_size[0];
    const int ny = fft_size[1];

    // Calculate eigenvalue factors
    m_eig_x.resize(nx);
    m_eig_y.resize(ny);
    amrex::Real * const eig_x_ptr = m_eig_x.dataPtr();
    amrex::Real * const eig_y_ptr = m_eig_y.dataPtr();

    const amrex::Real sine_x_factor = 1._rt / ( 2._rt * nx);
    const amrex::Real sine_y_factor = 1._rt / ( 2._rt * ny);
    const amrex::Real norm_fac = -4._rt * nx * ny;
    const amrex::Real invdxsq = gm.InvCellSize(0)*gm.InvCellSize(0)*norm_fac;
    const amrex::Real invdysq = gm.InvCellSize(1)*gm.InvCellSize(1)*norm_fac;

    amrex::ParallelFor(nx,
        [=] AMREX_GPU_DEVICE (int i) noexcept {
            const amrex::Real x_fac = amrex::Math::sinpi(sine_x_factor * (i+1));
            eig_x_ptr[i] = x_fac*x_fac*invdxsq;
        });

    amrex::ParallelFor(ny,
        [=] AMREX_GPU_DEVICE (int i) noexcept {
            const amrex::Real y_fac = amrex::Math::sinpi(sine_y_factor * (i+1));
            eig_y_ptr[i] = y_fac*y_fac*invdysq;
        });

    // Allocate 1d Array for 2d data or 2d transpose data
    m_real_array.resize(nx*ny);
    m_comp_array.resize(std::max((nx/2+1)*ny, (ny/2+1)*nx));

    // Allocate and initialize the FFT plans
    std::size_t fft1_area = m_x_r2cfft.Initialize(FFTType::R2C_1D_batched, nx, ny);
    std::size_t fft2_area = m_y_r2cfft.Initialize(FFTType::R2C_1D_batched, ny, nx);
    std::size_t fft3_area = m_x_c2rfft.Initialize(FFTType::C2R_1D_batched, nx, ny);
    std::size_t fft4_area = m_y_c2rfft.Initialize(FFTType::C2R_1D_batched, ny, nx);

    // Allocate work area for both FFTs
    m_fft_work_area.resize(std::max({fft1_area, fft2_area, fft3_area, fft4_area}));

    m_x_r2cfft.SetBuffers(m_real_array.dataPtr(), m_comp_array.dataPtr(),
                          m_fft_work_area.dataPtr());
    m_y_r2cfft.SetBuffers(m_real_array.dataPtr(), m_comp_array.dataPtr(),
                          m_fft_work_area.dataPtr());
    m_x_c2rfft.SetBuffers(m_comp_array.dataPtr(), m_real_array.dataPtr(),
                          m_fft_work_area.dataPtr());
    m_y_c2rfft.SetBuffers(m_comp_array.dataPtr(), m_real_array.dataPtr(),
                          m_fft_work_area.dataPtr());

    // Set up prefactors for dst2_out and dst3_in
    m_omega_x.resize(nx/2+1);
    amrex::GpuComplex<amrex::Real>* const omega_x_ptr = m_omega_x.dataPtr();
    amrex::ParallelFor(nx/2+1,
        [=] AMREX_GPU_DEVICE (int i) {
            auto [imag, real] = amrex::Math::sincospi(-i/amrex::Real(2._rt*nx));
            omega_x_ptr[i] = {real, imag};
        });

    m_omega_y.resize(ny/2+1);
    amrex::GpuComplex<amrex::Real>* const omega_y_ptr = m_omega_y.dataPtr();
    amrex::ParallelFor(ny/2+1,
        [=] AMREX_GPU_DEVICE (int i) {
            auto [imag, real] = amrex::Math::sincospi(-i/amrex::Real(2._rt*ny));
            omega_y_ptr[i] = {real, imag};
        });
}

void
FFTPoissonSolverDirichletQuick::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletQuick::SolvePoissonEquation()");

    const int nx = m_stagingArea[0].box().length(0); // initially contiguous
    const int ny = m_stagingArea[0].box().length(1); // contiguous after transpose

    Array2<amrex::Real> input_arr {{m_stagingArea[0].dataPtr(), {0,0,0}, {nx,ny,1}, 1}};

    Array2<amrex::Real> real_arr {{m_real_array.dataPtr(), {0,0,0}, {nx,ny,1}, 1}};
    Array2<amrex::Real> real_arr_t {{m_real_array.dataPtr(), {0,0,0}, {ny,nx,1}, 1}};

    Array2<amrex::GpuComplex<amrex::Real>> comp_arr {{m_comp_array.dataPtr(), {0,0,0}, {nx/2+1,ny,1}, 1}};
    Array2<amrex::GpuComplex<amrex::Real>> comp_arr_t {{m_comp_array.dataPtr(), {0,0,0}, {ny/2+1,nx,1}, 1}};

    amrex::Box lhs_bx = lhs_mf[0].box();
    // shift box to handle ghost cells properly
    lhs_bx -= m_stagingArea[0].box().smallEnd();
    Array2<amrex::Real> lhs_arr {{lhs_mf[0].dataPtr(), amrex::begin(lhs_bx), amrex::end(lhs_bx), 1}};

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            real_arr(i, j) = dst2_in(input_arr, i, j, nx);
        });

    m_x_r2cfft.Execute();

    dst2_out_t_in(comp_arr, real_arr_t, nx, ny, m_omega_x.dataPtr());

    m_y_r2cfft.Execute();

    dst2_out_mult_dst3_in(comp_arr_t, nx, ny, m_omega_y.dataPtr(), m_eig_x.dataPtr(), m_eig_y.dataPtr());

    m_y_c2rfft.Execute();

    dst3_out_t_in(real_arr_t, comp_arr, nx, ny, m_omega_x.dataPtr());

    m_x_c2rfft.Execute();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            lhs_arr(i, j) = dst3_out(real_arr, i, j, nx);
        });
}
