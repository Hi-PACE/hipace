#include "FFTPoissonSolver.H"
#include "Constants.H"

FFTPoissonSolver::FFTPoissonSolver ( amrex::BoxArray const& realspace_ba,
                                     amrex::DistributionMapping const& dm,
                                     amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}


void FFTPoissonSolver::define ( amrex::BoxArray const& realspace_ba,
                                amrex::DistributionMapping const& dm,
                                amrex::Geometry const& gm )
{
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
    m_spectralspace_ba.define( std::move(spectral_bl) );

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    m_stagingArea = amrex::MultiFab(realspace_ba, dm, 1, 0);
    m_tmpSpectralField = SpectralField(m_spectralspace_ba, dm, 1, 0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    // Calculate the array of inv_k2
    amrex::Real dkx = 2*MathConst::pi/gm.ProbLength(0);
    amrex::Real dky = 2*MathConst::pi/gm.ProbLength(1);
    m_inv_k2 = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);
    // Get the index of the local box in the slice BA.
    int box_id = m_inv_k2.IndexArray()[0];
    // Calculate inv_k2 in the current slice (currently, there is only 1 box per slice)
    amrex::Array4<amrex::Real> inv_k2_arr = m_inv_k2.array(box_id);
    amrex::Box const& bx = m_inv_k2.boxArray()[box_id];  // The lower corner of the "2D" slice Box is zero.
    int const Ny = bx.length(1);
    int const mid_point_y = (Ny+1)/2;
    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
        {
            // kx is always positive (first axis of the real-to-complex FFT)
            amrex::Real kx = dkx*i;
            // The first half of ky is positive ; the other is negative
            amrex::Real ky = (j<mid_point_y) ? dky*j : dky*(j-Ny);
            if ((i!=0) && (j!=0)) {
                inv_k2_arr(i,j,0) = 1./(kx*kx + ky*ky);
            } else {
                // Avoid division by 0
                inv_k2_arr(i,j,0) = 0;
            }
        });
    
    // Allocate and initialize the FFT plans
    m_forward_plan = AnyFFT::FFTplans(m_spectralspace_ba, dm);
    m_backward_plan = AnyFFT::FFTplans(m_spectralspace_ba, dm);
    // Allocate the plan for the box owned by the local MPI proc.
    // There can be only one on each rank, this is a fundamental assumption
    // that is NOT expected to change.
    box_id = m_stagingArea.IndexArray()[0];
    // Note: the size of the real-space box and spectral-space box
    // differ when using real-to-complex FFT. When initializing
    // the FFT plan, the valid dimensions are those of the real-space box.
    amrex::IntVect fft_size = m_stagingArea.boxArray()[box_id].length();
    m_forward_plan[box_id] = AnyFFT::CreatePlan(
        fft_size, m_stagingArea[box_id].dataPtr(),
        reinterpret_cast<AnyFFT::Complex*>( m_tmpSpectralField[box_id].dataPtr()),
        AnyFFT::direction::R2C);
    
    m_backward_plan[box_id] = AnyFFT::CreatePlan(
        fft_size, m_stagingArea[box_id].dataPtr(),
        reinterpret_cast<AnyFFT::Complex*>( m_tmpSpectralField[box_id].dataPtr()),
        AnyFFT::direction::C2R);
}


void
FFTPoissonSolver::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    // Get index of local box in the slice BA.
    int const box_id = m_stagingArea.IndexArray()[0];

    // Perform Fourier transform from the staging area to `tmpSpectralField`
    AnyFFT::Execute(m_forward_plan[box_id]);

    // Solve Poisson equation in Fourier space:
    // Multiply `tmpSpectralField` by inv_k2
    amrex::Array4<amrex::GpuComplex<amrex::Real>> tmp_cmplx_arr = m_tmpSpectralField.array(box_id);
    amrex::Array4<amrex::Real> inv_k2_arr = m_inv_k2.array(box_id);
    amrex::ParallelFor( m_spectralspace_ba[box_id],
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            tmp_cmplx_arr(i,j,k) *= inv_k2_arr(i,j,k);
                        });

    // Perform Fourier transform from `tmpSpectralField` to the staging area
    AnyFFT::Execute(m_backward_plan[box_id]);

    // Copy from the staging area to output array (and normalize)
    amrex::Array4<amrex::Real> tmp_real_arr = m_stagingArea.array(box_id);
    amrex::Array4<amrex::Real> lhs_arr = lhs_mf.array(box_id);
    amrex::Box const valid_box = m_stagingArea.boxArray()[m_stagingArea.IndexArray()[0]];
    const amrex::Real inv_N = 1./valid_box.numPts();
    amrex::ParallelFor( valid_box,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            // Copy and normalize field
                            lhs_arr(i,j,k) = inv_N*tmp_real_arr(i,j,k);
                        });
}
