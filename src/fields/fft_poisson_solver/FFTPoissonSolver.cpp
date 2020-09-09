#include "FFTPoissonSolver.H"

//FFTPoissonSolver::FFTPoissonSolver ( amrex::BoxArray const& realspace_ba,
//                                     amrex::DistributionMapping const& dm,
//                                     amrex::Geometry const& gm )
//{
//    define(realspace_ba, dm, gm);
//}

void
FFTPoissonSolver::define ( amrex::BoxArray const& /* realspace_ba */,
                           amrex::DistributionMapping const& /* dm */,
                           amrex::Geometry const& /* gm */)
{
    amrex::Abort("Should never hit that 1.");
}

void
FFTPoissonSolver::SolvePoissonEquation (amrex::MultiFab& /* lhs_mf */)
{
    amrex::Abort("Should never hit that 2.");    
}
