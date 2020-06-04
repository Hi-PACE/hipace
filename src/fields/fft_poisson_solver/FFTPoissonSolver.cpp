#include "FFTPoissonSolver.H"

FFTPoissonSolver::FFTPoissonSolver ()
{
    // initialize temporary multifabs
    // initialize FFT plans
    // initialize k space inv_k2
}

void
FFTPoissonSolver::SolvePoissonEquation ( amrex::MultiFab const input_mf,
                                         amrex::MultiFab output_mf )
{
    // Copy to temporary
    // Perform FFT
    // Divide by k2
    // Perform inverse FFT
    // Copy from temporary to output array
}
