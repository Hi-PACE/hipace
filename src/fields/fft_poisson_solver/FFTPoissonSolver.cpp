#include "FFTPoissonSolver.H"

FFTPoissonSolver::~FFTPoissonSolver ()
{}

amrex::MultiFab&
FFTPoissonSolver::StagingArea ()
{
    return m_stagingArea;
}
