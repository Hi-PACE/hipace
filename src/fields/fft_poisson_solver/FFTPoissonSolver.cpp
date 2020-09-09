#include "FFTPoissonSolver.H"

amrex::MultiFab&
FFTPoissonSolver::StagingArea ()
{
    return m_stagingArea;
}
