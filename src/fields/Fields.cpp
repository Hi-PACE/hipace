#include "Fields.H"

void
Fields::AllocData (int lev, const amrex::BoxArray& ba,
                   const amrex::DistributionMapping& dm, amrex::Geometry& geom)
{
    m_F[lev] = amrex::MultiFab(ba, dm, FieldComps::nfields, m_nguards);
    for (int islice=0; islice<m_nslices; islice++){
        // The domain for slices is the whole domain transversally, and only 1 cell longitudinally.
        amrex::Box slice_domain = geom.Domain();
        slice_domain.setSmall(AMREX_SPACEDIM-1, 0.);
        slice_domain.setBig(AMREX_SPACEDIM-1, 0.);
        // WARNING: If transverse parallelization, slice_ba and slice_dm are currently independent
        // from ba and dm, which has to be changed for performance.
        amrex::BoxArray slice_ba(slice_domain);
        amrex::DistributionMapping slice_dm {slice_ba};
        m_slices[lev][islice] = new amrex::MultiFab(slice_ba, slice_dm, FieldComps::nfields, m_slices_nguards);
    }
}

void
Fields::TransverseDerivative(const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                             const amrex::Real dx, const int scomp, const int dcomp)
{
    AMREX_ALWAYS_ASSERT((direction == 0) || (direction == 1));
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (direction == 0){
                    /* finite difference along x */
                    dst_array(i,j,k,dcomp) =
                        (src_array(i+1, j, k, scomp) - src_array(i-1, j, k, scomp)) / (2*dx);
                } else {
                    /* finite difference along y */
                    dst_array(i,j,k,dcomp) =
                        (src_array(i, j+1, k, scomp) - src_array(i, j-1, k, scomp)) / (2*dx);
                }
            }
            );
    }
}
