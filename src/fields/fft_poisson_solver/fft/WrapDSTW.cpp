#include "AnyDST.H"
#include "utils/HipaceProfilerWrapper.H"

namespace AnyDST
{
#ifdef AMREX_USE_FLOAT
    const auto VendorCreatePlanR2R2D = fftwf_plan_r2r_2d;
#else
    const auto VendorCreatePlanR2R2D = fftw_plan_r2r_2d;
#endif

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        DSTplan dst_plan;
        const int nx = real_size[0];
        const int ny = real_size[1];

        // Initialize fft_plan.m_plan with the vendor fft plan.
        // Swap dimensions: AMReX FAB are Fortran-order but FFTW is C-order
        dst_plan.m_plan = VendorCreatePlanR2R2D(
            ny, nx, position_array->dataPtr(), fourier_array->dataPtr(),
            FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);

        // Initialize fft_plan.m_plan_b with the vendor fft plan.
        // Swap arrays: now for backward direction.
        dst_plan.m_plan_b = VendorCreatePlanR2R2D(
            ny, nx, fourier_array->dataPtr(), position_array->dataPtr(),
            FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);

        // Store meta-data in fft_plan
        dst_plan.m_position_array = position_array;
        dst_plan.m_fourier_array = fourier_array;

        return dst_plan;
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
#  ifdef AMREX_USE_FLOAT
        fftwf_destroy_plan( dst_plan.m_plan );
        fftwf_destroy_plan( dst_plan.m_plan_b );
#  else
        fftw_destroy_plan( dst_plan.m_plan );
        fftw_destroy_plan( dst_plan.m_plan_b );
#  endif
    }

    template<direction d>
    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");
        // Swap position and fourier space based on execute direction
        AnyFFT::VendorFFTPlan& plan = (d==direction::forward) ? dst_plan.m_plan : dst_plan.m_plan_b;
#  ifdef AMREX_USE_FLOAT
        fftwf_execute( plan );
#  else
        fftw_execute( plan );
#  endif
    }

    template void Execute<direction::forward>(DSTplan& dst_plan);
    template void Execute<direction::backward>(DSTplan& dst_plan);

}
