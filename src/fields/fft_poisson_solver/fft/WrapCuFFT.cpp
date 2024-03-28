/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Remi Lehe, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "AnyFFT.H"
#include "CuFFTUtils.H"
#include "utils/HipaceProfilerWrapper.H"

namespace AnyFFT
{

#ifdef AMREX_USE_FLOAT
    cufftType VendorR2C = CUFFT_R2C;
    cufftType VendorC2R = CUFFT_C2R;
#else
    cufftType VendorR2C = CUFFT_D2Z;
    cufftType VendorC2R = CUFFT_Z2D;
#endif

    FFTplan CreatePlan (const amrex::IntVect& real_size, amrex::Real * const real_array,
                        Complex * const complex_array, const direction dir)
    {
        HIPACE_PROFILE("AnyFFT::CreatePlan()");
        FFTplan fft_plan;

        if (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 1) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE((std::max(real_size[0], real_size[1]) <= 1024),
            "Due to a bug in cuFFT, CUDA 11.1 supports only nx, ny <= 1024. Please use CUDA "
            "version >= 11.2 (recommended) or <= 11.0 for larger grid sizes.");
        }

        // Initialize fft_plan.m_plan with the vendor fft plan.
        cufftResult result;
        if (dir == direction::R2C){
            result = cufftPlan2d(
                &(fft_plan.m_plan), real_size[1], real_size[0], VendorR2C);
        } else {
            result = cufftPlan2d(
                &(fft_plan.m_plan), real_size[1], real_size[0], VendorC2R);
        }

        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " cufftplan failed! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }

        // Store meta-data in fft_plan
        fft_plan.m_real_array = real_array;
        fft_plan.m_complex_array = complex_array;
        fft_plan.m_dir = dir;

        return fft_plan;
    }

    void DestroyPlan (FFTplan& fft_plan)
    {
        cufftDestroy( fft_plan.m_plan );
    }

    void Execute (FFTplan& fft_plan){
        HIPACE_PROFILE("AnyFFT::Execute()");
        // make sure that this is done on the same GPU stream as the above copy
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream ( fft_plan.m_plan, stream);
        cufftResult result;
        if (fft_plan.m_dir == direction::R2C){
#ifdef AMREX_USE_FLOAT
            result = cufftExecR2C(fft_plan.m_plan, fft_plan.m_real_array, fft_plan.m_complex_array);
#else
            result = cufftExecD2Z(fft_plan.m_plan, fft_plan.m_real_array, fft_plan.m_complex_array);
#endif
        } else if (fft_plan.m_dir == direction::C2R){
#ifdef AMREX_USE_FLOAT
            result = cufftExecC2R(fft_plan.m_plan, fft_plan.m_complex_array, fft_plan.m_real_array);
#else
            result = cufftExecZ2D(fft_plan.m_plan, fft_plan.m_complex_array, fft_plan.m_real_array);
#endif
        } else {
            amrex::Abort("direction must be AnyFFT::direction::R2C or AnyFFT::direction::C2R");
        }
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
    }
}
