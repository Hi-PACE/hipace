#include "FFTPoissonSolver.H"

FFTPoissonSolver::FFTPoissonSolver ( amrex::BoxArray const& realspace_ba,
                                     amrex::DistributionMapping const& dm )
{
    // Create the box array that corresponds to spectral space
    amrex::BoxList spectral_bl; // Create empty box list
    // Loop over boxes and fill the box list
    for (int i=0; i < realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        amrex::Box realspace_bx = realspace_ba[i];
        amrex::IntVect fft_size = realspace_bx.length();
        // Because the spectral solver uses real-to-complex FFTs, we only
        // need the positive k values along the fastest axis
        // (first axis for AMReX Fortran-order arrays) in spectral space.
        // This effectively reduces the size of the spectral space by half
        // see e.g. the FFTW documentation for real-to-complex FFTs
        amrex::IntVect spectral_bx_size = fft_size;
        spectral_bx_size[0] = fft_size[0]/2 + 1;
        // Define the corresponding box
        amrex::Box spectral_bx = amrex::Box( amrex::IntVect::TheZeroVector(),
                          spectral_bx_size - amrex::IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
    }
    m_spectralspace_ba.define( spectral_bl );

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    m_tmpRealField = amrex::MultiFab(realspace_ba, dm, 1, 0);
    m_tmpSpectralField = SpectralField(m_spectralspace_ba, dm, 1, 0);

    // Allocate and initialize the FFT plans
    m_forward_plan = AnyFFT::FFTplans(m_spectralspace_ba, dm);
    m_backward_plan = AnyFFT::FFTplans(m_spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( amrex::MFIter mfi(m_spectralspace_ba, dm); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        amrex::IntVect fft_size = realspace_ba[mfi].length();

        m_forward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, m_tmpRealField[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( m_tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::R2C);

        m_backward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, m_tmpRealField[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( m_tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::C2R);
    }
}


void
FFTPoissonSolver::SolvePoissonEquation ( amrex::MultiFab const& input_mf,
                                         amrex::MultiFab& output_mf )
{
    // Copy to temporary
    // Perform FFT
    // Divide by k2
    // Perform inverse FFT
    // Copy from temporary to output array
}
