#include "Hipace.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

Hipace::Hipace () :
    m_fields(amrex::AmrCore::maxLevel())
{
    amrex::ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
    pp.query("max_step", m_max_step);
}

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

void
Hipace::Evolve ()
{
    WriteDiagnostics (0);
    for (int step; step < m_max_step; ++step)
    {
        amrex::Print()<<"step "<< step <<"\n";
    }
    WriteDiagnostics (1);
}

void
Hipace::WriteDiagnostics (int step)
{
    const std::string filename = amrex::Concatenate("plt", step);
    const int nlev = 1;
    const amrex::Vector< std::string > varnames {"Ex", "Ey", "Ez"};
    const int time = 0.;
    const amrex::IntVect ref_ratio {1, 1, 1};
    amrex::Vector<std::string> rfs;
    amrex::WriteMultiLevelPlotfile(filename, nlev,
                                   amrex::GetVecOfConstPtrs(m_fields.getF()),
                                   varnames, Geom(),
                                   time, {step}, {ref_ratio},
                                   "HyperCLaw-V1.1",
                                   "Level_",
                                   "Cell",
                                   rfs
                                   );
}
