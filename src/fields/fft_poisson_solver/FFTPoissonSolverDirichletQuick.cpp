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
    if (i == n-1) {
        return in(0, j).real();
    } else if (2*i+1 < n) {
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
    m_position_array2.resize(nx*ny);
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

    Array2<amrex::Real> real_arr2 {{m_position_array2.dataPtr(), {0,0,0}, {nx,ny,1}, 1}};
    Array2<amrex::Real> real_arr2_t {{m_position_array2.dataPtr(), {0,0,0}, {ny,nx,1}, 1}};

    Array2<amrex::GpuComplex<amrex::Real>> comp_arr {{ m_fourier_array.dataPtr(), {0,0,0}, {nx/2+1,ny,1}, 1}};
    Array2<amrex::GpuComplex<amrex::Real>> comp_arr_t {{ m_fourier_array.dataPtr(), {0,0,0}, {ny/2+1,nx,1}, 1}};

    amrex::Box lhs_bx = lhs_mf[0].box();
    // shift box to handle ghost cells properly
    lhs_bx -= m_stagingArea[0].box().smallEnd();
    Array2<amrex::Real> lhs_arr {{lhs_mf[0].dataPtr(), amrex::begin(lhs_bx), amrex::end(lhs_bx), 1}};

    Array2<amrex::Real> mult_t {m_eigenvalue_matrix.array()};
    const amrex::GpuComplex<amrex::Real>* omage_x_ptr = m_omega_x.dataPtr();
    const amrex::GpuComplex<amrex::Real>* omage_y_ptr = m_omega_y.dataPtr();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            real_arr(i, j) = dst2_in(pos_arr, i, j, nx);
        });

    m_x_r2cfft.Execute();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            real_arr(i, j) = dst2_out(comp_arr, i, j, nx, omage_x_ptr);
        });

    amrex::transposeCtoF(real_arr.p, real_arr2_t.p, ny, nx);

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {ny-1, nx-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            real_arr_t(i, j) = dst2_in(real_arr2_t, i, j, ny);
        });

    m_y_r2cfft.Execute();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {ny/2, nx-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            dst2_out_mult_dst3_in(comp_arr_t, mult_t, i, j, ny, omage_y_ptr);
        });

    m_y_c2rfft.Execute();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {ny-1, nx-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            real_arr2_t(i, j) = dst3_out(real_arr_t, i, j, ny);
        });

    amrex::transposeCtoF(real_arr2_t.p, real_arr.p, nx, ny);

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx/2, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            comp_arr(i, j) = dst3_in(real_arr, i, j, nx, omage_x_ptr);
        });

    m_x_c2rfft.Execute();

    amrex::ParallelFor(amrex::BoxND<2>{{0, 0}, {nx-1, ny-1}},
        [=] AMREX_GPU_DEVICE (int i, int j){
            lhs_arr(i, j) = dst3_out(real_arr, i, j, nx);
        });
}
