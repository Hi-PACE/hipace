#include "AnyDST.H"
#include "HipaceProfilerWrapper.H"

namespace AnyDST
{
#ifdef AMREX_USE_FLOAT
    const auto VendorCreatePlanR2R2D = fftwf_plan_r2r_2d;
#else
    const auto VendorCreatePlanR2R2D = fftw_plan_r2r_2d;
#endif

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::Real * const position_array,
                        amrex::Real * const fourier_array)
    {
        DSTplan dst_plan;

        // Initialize fft_plan.m_plan with the vendor fft plan.
        // Swap dimensions: AMReX FAB are Fortran-order but FFTW is C-order
        dst_plan.m_plan = fftw_plan_r2r_2d(
            real_size[1], real_size[0], position_array, fourier_array,
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
#  else
        fftw_destroy_plan( dst_plan.m_plan );
#  endif
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("Execute_FFTplan()");
#  ifdef AMREX_USE_FLOAT
        fftwf_execute( dst_plan.m_plan );
#  else
        fftw_execute( dst_plan.m_plan );
#  endif
    }

}
