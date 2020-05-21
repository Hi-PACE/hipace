#include "Hipace.H"

Hipace::Hipace () :
    m_fields(amrex::AmrCore::maxLevel())
{}

void
Hipace::InitData ()
{
    AmrCore::InitFromScratch(0.0); // function argument is time
}

void
Hipace::MakeNewLevelFromScratch (
    int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    AMREX_ALWAYS_ASSERT(lev == 0);
    m_fields.AllocData(lev, ba, dm);
}
