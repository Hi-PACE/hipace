#include "AnyDST.H"
#include "utils/HipaceProfilerWrapper.H"

#if defined(AMREX_USE_HIP)
#   include "rocfft.h"
#elif defined(AMREX_USE_CUDA)
#   include "CuFFTUtils.H"
# endif

namespace AnyDST
{

#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
    /** \brief Extend src into a symmetrized larger array dst
     *
     * \param[in,out] dst destination array, odd symmetry around 0 and the middle points in x and y
     * \param[in] src source array
     */
    void ExpandR2R (amrex::FArrayBox& dst, amrex::FArrayBox& src)
    {
        HIPACE_PROFILE("AnyDST::ExpandR2R()");
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = src.box();
        const int nx = bx.length(0);
        const int ny = bx.length(1);
        amrex::Array4<amrex::Real const> const & src_array = src.array();
        amrex::Array4<amrex::Real> const & dst_array = dst.array();

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i+1,j+1,0,dcomp) = src_array(i, j, k, scomp);
                /* lower left quadrant */
                dst_array(i+1,j+ny+2,0,dcomp) = -src_array(i, ny-1-j, k, scomp);
                /* upper right quadrant */
                dst_array(i+nx+2,j+1,0,dcomp) = -src_array(nx-1-i, j, k, scomp);
                /* lower right quadrant */
                dst_array(i+nx+2,j+ny+2,0,dcomp) = src_array(nx-1-i, ny-1-j, k, scomp);
            }
            );
    };

    /** \brief Extract symmetrical src array into smaller array dst
     *
     * \param[in,out] dst destination array
     * \param[in] src destination array, symmetric in x and y
     */
    void ShrinkC2R (amrex::FArrayBox& dst, amrex::BaseFab<amrex::GpuComplex<amrex::Real>>& src)
    {
        HIPACE_PROFILE("AnyDST::ShrinkC2R()");
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = dst.box();
        amrex::Array4<amrex::GpuComplex<amrex::Real> const> const & src_array = src.array();
        amrex::Array4<amrex::Real> const & dst_array = dst.array();
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i,j,k,dcomp) = -src_array(i+1, j+1, 0, scomp).real();
            }
            );
    };
#endif //expand and shrink only for GPU frameworks

#if defined(AMREX_USE_HIP)
#   include "WrapRocDST.h"
#elif defined(AMREX_USE_CUDA)
#   include "WrapCuDST.h"
#else
#   include "WrapDSTW.h
#endif


}
