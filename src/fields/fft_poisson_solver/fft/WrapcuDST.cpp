#include "AnyDST.H"
#include "HipaceProfilerWrapper.H"

namespace AnyDST
{
    void ExpandR2R(amrex::Real * const dst, amrex::Real const * const src)
    {
        // --- Expand src to dst
    };

    void ShrinkC2R(amrex::Real * const dst, AnyFFT::Complex const * const src)
    {
        // --- Shrink src to dst
    };

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::Real * const position_array,
                        amrex::Real * const fourier_array)
    {
        DSTplan dst_plan;

        // --- Allocate expanded_array Real of size (2*nx+2, 2*ny+2)
        amrex::Real* expanded_array = position_array; // THIS IS NOT CORRECT

        const amrex::IntVect& expanded_size {2*real_size[1]+2, 2*real_size[0]+2, 1};

        // Initialize fft_plan.m_plan with the vendor fft plan.
        cufftResult result;
        result = cufftPlan2d(
            &(dst_plan.m_plan), expanded_size[1], expanded_size[0], VendorR2C);

        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " cufftplan failed! Error: " <<
                cufftErrorToString(result) << "\n";
        }

        // Store meta-data in dst_plan
        dst_plan.m_position_array = position_array;
        dst_plan.m_expanded_array = expanded_array;
        dst_plan.m_fourier_array = fourier_array;

        return dst_plan;
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        cufftDestroy( dst_plan.m_plan );
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("Execute_DSTplan()");

        ExpandR2R(dst_plan.m_expanded_array, dst_plan.m_position_array);

        // make sure that this is done on the same GPU stream as the above copy
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream ( dst_plan.m_plan, stream);
        cufftResult result;
#ifdef AMREX_USE_FLOAT
        result = cufftExecR2C(
            dst_plan.m_plan, dst_plan.m_expanded_array, dst_plan.m_complex_array);
#else
        result = cufftExecD2Z(
            dst_plan.m_plan, dst_plan.m_expanded_array, dst_plan.m_complex_array);
#endif

        ShrinkC2R(dst_plan.m_position_array, dst_plan.m_fourier_array);
    }    
}
