#include "FFTPoissonSolverUtil.H"
#include "HipaceProfilerWrapper.H"
#include <AMReX_GpuComplex.H>

void
ExpandRealData (const amrex::MultiFab& src, amrex::MultiFab& dst, amrex::Geometry const& geom,
                const int scomp, const int dcomp)
{
    HIPACE_PROFILE("ExpandRealData");
    /* This function takes an nx*ny grid point multifab src as input and returns
     * a (2nx+2)*(2ny+2) multifab dst, where the data of src is symmetrized, so
     * dst is odd around 0 and the midpoint in both dimensions.
     * This is necessary  before a 2D discrete fourtier transform to achieve the same result
     * as a 2D discrete sine transform */

    const int nx = geom.Domain().length(0);
    const int ny = geom.Domain().length(1);

    for ( amrex::MFIter mfi(src, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
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
    }
}

void
ShrinkRealData (const amrex::MultiFab& src, amrex::MultiFab& dst, amrex::Geometry const& geom,
                const int scomp, const int dcomp)
{
    HIPACE_PROFILE("ShrinkRealData");
    /* This function takes an (2nx+2)*(2ny+2) grid point multifab src as input and returns
     * an nx*ny multifab dst, where the data from the src is extracted assuming symmetry.
     * This is necessary after a 2D discrete fourtier transform to achieve the same result
     *  as a 2D discrete sine transform */

    const int nx = geom.Domain().length(0);
    const int ny = geom.Domain().length(1);

    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i,j,k,dcomp) = -src_array(i+1, j+1, k, scomp);
            }
            );
    }
}
