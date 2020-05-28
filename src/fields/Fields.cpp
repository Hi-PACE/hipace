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
