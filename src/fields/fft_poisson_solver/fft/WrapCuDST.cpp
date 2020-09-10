#include "AnyDST.H"
#include "HipaceProfilerWrapper.H"

namespace AnyDST
{
    void ExpandR2R (amrex::FArrayBox& dst, amrex::FArrayBox& src)
    {
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = src.box();
        amrex::Print()<<bx<<"  -- expand \n";
        const int nx = bx.length(0);
        const int ny = bx.length(1);
        amrex::Array4<amrex::Real const> const & src_array = src.array();
        amrex::Array4<amrex::Real> const & dst_array = dst.array();

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i+1,j+1,k,dcomp) = src_array(i, j, k, scomp);
                /* lower left quadrant */
                dst_array(i+1,j+ny+2,k,dcomp) = -src_array(i, ny-1-j, k, scomp);
                /* upper right quadrant */
                dst_array(i+nx+2,j+1,k,dcomp) = -src_array(nx-1-i, j, k, scomp);
                /* lower right quadrant */
                dst_array(i+nx+2,j+ny+2,k,dcomp) = src_array(nx-1-i, ny-1-j, k, scomp);
            }
            );
    };

    void ShrinkC2R (amrex::FArrayBox& dst, amrex::BaseFab<amrex::GpuComplex<amrex::Real>>& src)
    {
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = dst.box();
        amrex::Print()<<bx<<"  -- shrink \n";
        amrex::Array4<amrex::GpuComplex<amrex::Real> const> const & src_array = src.array();
        amrex::Array4<amrex::Real> const & dst_array = dst.array();
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i,j,k,dcomp) = -src_array(i+1, j+1, k, scomp).real();
            }
            );
    };

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        DSTplan dst_plan;
        const int nx = real_size[0];
        const int ny = real_size[1];

        // Allocate expanded_position_array Real of size (2*nx+2, 2*ny+2)
        // Allocate expanded_fourier_array Complex of size (nx+1, 2*ny+2)
        amrex::Box expanded_position_box {{0, 0, 0}, {2*nx+1, 2*ny+1, 0}};
        amrex::Box expanded_fourier_box {{0, 0, 0}, {nx, 2*ny+1, 0}};
        dst_plan.m_expanded_position_array =std::make_unique<
            amrex::FArrayBox>(expanded_position_box, 1);
        dst_plan.m_expanded_fourier_array = std::make_unique<
            amrex::BaseFab<amrex::GpuComplex<amrex::Real>>>(expanded_fourier_box, 1);

        const amrex::IntVect& expanded_size = expanded_position_box.length();

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
        dst_plan.m_fourier_array = fourier_array;

        return dst_plan;
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        cufftDestroy( dst_plan.m_plan );
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("Execute_DSTplan()");

        // Expand in position space m_position_array -> m_expanded_position_array
        ExpandR2R(dst_plan.m_expanded_position_array, dst_plan.m_position_array);

        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream ( dst_plan.m_plan, stream);
        cufftResult result;

        // R2C FFT m_expanded_position_array -> m_expanded_fourier_array
#ifdef AMREX_USE_FLOAT
        result = cufftExecR2C(
            dst_plan.m_plan, dst_plan.m_expanded_position_array, dst_plan.m_expanded_fourier_array);
#else
        result = cufftExecD2Z(
            dst_plan.m_plan, dst_plan.m_expanded_position_array, dst_plan.m_expanded_fourier_array);
#endif

        // Shrink in Fourier space m_expanded_fourier_array -> m_fourier_array
        ShrinkC2R(dst_plan.m_fourier_array, dst_plan.m_expanded_fourier_array);
    }
}
