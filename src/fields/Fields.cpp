#include "Fields.H"

void
Fields::AllocData (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    m_F[lev] = amrex::MultiFab(ba, dm, FieldComps::nfields, m_nguards);
}
