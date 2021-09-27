#include "FFTPoissonSolverDirichlet.H"
#include "fft/AnyDST.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichlet::FFTPoissonSolverDirichlet (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

void
FFTPoissonSolverDirichlet::define (amrex::BoxArray const& a_realspace_ba,
                                   amrex::DistributionMapping const& dm,
                                   amrex::Geometry const& gm )
{
    using namespace amrex::literals;

    HIPACE_PROFILE("FFTPoissonSolverDirichlet::define()");
    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_realspace_ba.size() == 1, "Parallel FFT not supported yet");

    // Create the box array that corresponds to spectral space
    amrex::BoxList spectral_bl; // Create empty box list
    amrex::BoxList real_bl; // Create empty box list<
    // Loop over boxes and fill the box list
    for (int i=0; i < a_realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        // Define the corresponding box
        amrex::Box spectral_bx = amrex::Box( amrex::IntVect::TheZeroVector(),
                          a_realspace_ba[i].length() - amrex::IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
        amrex::Box real_bx = spectral_bx;
        real_bx.setSmall(Direction::z, a_realspace_ba[i].smallEnd(Direction::z));
        real_bx.setBig  (Direction::z, a_realspace_ba[i].bigEnd(Direction::z));
        real_bl.push_back( real_bx );
    }
    m_spectralspace_ba.define( std::move(spectral_bl) );
    amrex::BoxArray real_ba(std::move(real_bl));

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    // The stagingArea is also created from 0 to nx, because the real space array may have
    // an offset for levels > 0
    m_stagingArea = amrex::MultiFab(real_ba, dm, 1, 0);
    m_tmpSpectralField = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);
    m_stagingArea.setVal(0.0); // this is not required
    m_tmpSpectralField.setVal(0.0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    const auto dx = gm.CellSizeArray();
    const amrex::Real dxsquared = dx[0]*dx[0];
    const amrex::Real dysquared = dx[1]*dx[1];
    const amrex::Real sine_x_factor = MathConst::pi / ( 2. * ( real_ba[0].length(0) + 1 ));
    const amrex::Real sine_y_factor = MathConst::pi / ( 2. * ( real_ba[0].length(1) + 1 ));

    // Normalization of FFTW's 'DST-I' discrete sine transform (FFTW_RODFT00)
    // This normalization is used regardless of the sine transform library
    const amrex::Real norm_fac = 0.5 / ( 2 * (( real_ba[0].length(0) + 1 )
                                             *( real_ba[0].length(1) + 1 )));

    m_eigenvalue_matrix = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);

    // Calculate the array of m_eigenvalue_matrix
    for (amrex::MFIter mfi(m_eigenvalue_matrix); mfi.isValid(); ++mfi ){
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);
        amrex::Box const& bx = mfi.validbox();  // The lower corner of the "2D" slice Box is zero.
        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
                {
                    /* fast poisson solver diagonal x coeffs */
                    amrex::Real sinex_sq = sin(( i + 1 ) * sine_x_factor) * sin(( i + 1 ) * sine_x_factor);
                    /* fast poisson solver diagonal y coeffs */
                    amrex::Real siney_sq = sin(( j + 1 ) * sine_y_factor) * sin(( j + 1 ) * sine_y_factor);

                    if ((sinex_sq!=0) && (siney_sq!=0)) {
                        eigenvalue_matrix(i,j,0) = norm_fac / ( -4.0 * ( sinex_sq / dxsquared + siney_sq / dysquared ));
                    } else {
                        // Avoid division by 0
                        eigenvalue_matrix(i,j,0) = 0._rt;
                    }
                });
    }

    // Allocate and initialize the FFT plans
    m_plan = AnyDST::DSTplans(m_spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        amrex::IntVect fft_size = mfi.validbox().length();
        m_plan[mfi] = AnyDST::CreatePlan(
            fft_size, &m_stagingArea[mfi], &m_tmpSpectralField[mfi]);
    }
}


void
FFTPoissonSolverDirichlet::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::SolvePoissonEquation()");

    // Loop over boxes
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){

        // Perform Fourier transform from the staging area to `tmpSpectralField`
        AnyDST::Execute<AnyDST::direction::forward>(m_plan[mfi]);

        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by eigenvalue_matrix
        amrex::Array4<amrex::Real> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);

        amrex::ParallelFor( m_spectralspace_ba[mfi],
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_cmplx_arr(i,j,k) *= eigenvalue_matrix(i,j,k);
            });

        // Perform Fourier transform from `tmpSpectralField` to the staging area
        AnyDST::Execute<AnyDST::direction::backward>(m_plan[mfi]);

        // Copy from the staging area to output array (and normalize)
        amrex::Array4<amrex::Real> tmp_real_arr = m_stagingArea.array(mfi);
        amrex::Array4<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lhs_mf.size() == 1,
                                         "Slice MFs must be defined on one box only");
        const amrex::FArrayBox& lhs_fab = lhs_mf[0];
        amrex::Box lhs_bx = lhs_fab.box();
        lhs_bx.grow({-Fields::m_slices_nguards[0], -Fields::m_slices_nguards[1], 0});
        const amrex::IntVect lo = lhs_bx.smallEnd();
        amrex::ParallelFor( mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Copy and normalize field
                lhs_arr(i+lo[0],j+lo[1],k) = tmp_real_arr(i,j,k);
            });
    }
}
