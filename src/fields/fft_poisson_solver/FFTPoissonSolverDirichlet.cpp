#include "FFTPoissonSolverDirichlet.H"
#include "Constants.H"
#include "HipaceProfilerWrapper.H"

FFTPoissonSolverDirichlet::FFTPoissonSolverDirichlet (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

void
FFTPoissonSolverDirichlet::define ( amrex::BoxArray const& realspace_ba,
                                   amrex::DistributionMapping const& dm,
                                   amrex::Geometry const& gm )
{
    using namespace amrex::literals;

    HIPACE_PROFILE("FFTPoissonSolverDirichlet::define()");
    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(realspace_ba.size() == 1, "Parallel FFT not supported yet");

    // Create the box array that corresponds to spectral space
    amrex::BoxList spectral_bl; // Create empty box list
    // Loop over boxes and fill the box list
    for (int i=0; i < realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        amrex::Box realspace_bx = realspace_ba[i];
        amrex::IntVect fft_size = realspace_bx.length();
        // Define the corresponding box
        amrex::Box spectral_bx = amrex::Box( amrex::IntVect::TheZeroVector(),
                          fft_size - amrex::IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
    }
    m_spectralspace_ba.define( std::move(spectral_bl) );

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    m_stagingArea = amrex::MultiFab(realspace_ba, dm, 1, 0);
    m_tmpSpectralField = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    // Calculate the array of m_eigenvalue_matrix
    const auto dx = gm.CellSizeArray();
    const amrex::Real dxsquared = dx[0]*dx[0];
    const amrex::Real dysquared = dx[1]*dx[1];
    const amrex::Real sine_x_factor = MathConst::pi / ( 2. * ( gm.Domain().length(0) + 1 ));
    const amrex::Real sine_y_factor = MathConst::pi / ( 2. * ( gm.Domain().length(1) + 1 ));

    /* Normalization of FFTW's 'DST-I' discrete sine transform (FFTW_RODFT00) */
    const amrex::Real norm_fac = 0.5 / ( 2 * (( gm.Domain().length(0) + 1 ) * ( gm.Domain().length(1) + 1 )));

    m_eigenvalue_matrix = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);

// Loop over boxes and calculate inv_k2 in each box
    for (amrex::MFIter mfi(m_eigenvalue_matrix); mfi.isValid(); ++mfi ){
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);
        amrex::Box const& bx = mfi.validbox();  // The lower corner of the "2D" slice Box is zero.
        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
                {
                    /* fast poisson solver diagonal x coeffs */
                    amrex::Real sinex_sq = pow( sin(( i + 1 ) * sine_x_factor), 2);
                    /* fast poisson solver diagonal y coeffs */
                    amrex::Real siney_sq = pow( sin(( j + 1 ) * sine_y_factor), 2);

                    if ((sinex_sq!=0) && (siney_sq!=0)) {
                        eigenvalue_matrix(i,j,0) = norm_fac / ( -4.0 * ( sinex_sq / dxsquared + siney_sq / dysquared ));
                    } else {
                        // Avoid division by 0
                        eigenvalue_matrix(i,j,0) = 0._rt;
                    }
                });
    }

    // Allocate and initialize the FFT plans
    m_forward_plan = AnyDST::DSTplans(m_spectralspace_ba, dm);
    m_backward_plan = AnyDST::DSTplans(m_spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        amrex::IntVect fft_size = mfi.validbox().length();
        m_forward_plan[mfi] = AnyDST::CreatePlan(
            fft_size, m_stagingArea[mfi].dataPtr(), m_tmpSpectralField[mfi].dataPtr());

        m_backward_plan[mfi] = AnyDST::CreatePlan(
            fft_size, m_tmpSpectralField[mfi].dataPtr(), m_stagingArea[mfi].dataPtr());
    }
}


void
FFTPoissonSolverDirichlet::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::SolvePoissonEquation()");

    // Loop over boxes
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){

        // Perform Fourier transform from the staging area to `tmpSpectralField`
        AnyDST::Execute(m_forward_plan[mfi]);

        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by eigenvalue_matrix
        amrex::Array4<amrex::Real> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);

        amrex::ParallelFor( m_spectralspace_ba[mfi],
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_cmplx_arr(i,j,k) *= eigenvalue_matrix(i,j,k);
            });

        // Perform Fourier transform from `tmpSpectralField` to the staging area
        AnyDST::Execute(m_backward_plan[mfi]);

        // Copy from the staging area to output array (and normalize)
        amrex::Array4<amrex::Real> tmp_real_arr = m_stagingArea.array(mfi);
        amrex::Array4<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        // const amrex::Real inv_N = 1./mfi.validbox().numPts();
        amrex::ParallelFor( mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Copy and normalize field
                // lhs_arr(i,j,k) = inv_N*tmp_real_arr(i,j,k);
                lhs_arr(i,j,k) = tmp_real_arr(i,j,k);
            });

    }
}
