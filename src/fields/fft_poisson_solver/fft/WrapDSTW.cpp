#include "AnyDST.H"

namespace AnyDST
{
#ifdef AMREX_USE_FLOAT
    const auto VendorCreatePlanR2R2D = fftwf_plan_r2r_2d;
#else
    const auto VendorCreatePlanR2R2D = fftw_plan_r2r_2d;
#endif
    
    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::Real * const position_array,
                        amrex::Real * const fourier_array, const direction dir)
    {
        DSTplan dst_plan;

        // Initialize fft_plan.m_plan with the vendor fft plan.
        // Swap dimensions: AMReX FAB are Fortran-order but FFTW is C-order
        dst_plan.m_plan = fftw_plan_r2r_2d(
            real_size[1], real_size[0], real_array, complex_array,
            FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);

        // Store meta-data in fft_plan
        fft_plan.m_position_array = position_array;
        fft_plan.m_fourier_array = fourier_array;

        return fft_plan;
    }

    void DestroyPlan (DSTplan& fft_plan)
    {
#  ifdef AMREX_USE_FLOAT
        fftwf_destroy_plan( fft_plan.m_plan );
#  else
        fftw_destroy_plan( fft_plan.m_plan );
#  endif
    }

    void Execute (DSTplan& fft_plan){
        HIPACE_PROFILE("Execute_FFTplan()");
#  ifdef AMREX_USE_FLOAT
        fftwf_execute( fft_plan.m_plan );
#  else
        fftw_execute( fft_plan.m_plan );
#  endif
    }

}
