#ifndef LASER_H_
#define LASER_H_

#include "fields/Fields.H"
#include "mg_solver/HpMultiGrid.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>
#include <AMReX_GpuComplex.H>

#ifdef AMREX_USE_CUDA
#  include <cufft.h>
#elif defined(AMREX_USE_HIP)
#  include <cstddef>
#  include <rocfft.h>
#else
#  include <fftw3.h>
#endif

namespace LaserFFT {
#ifdef AMREX_USE_CUDA
    using VendorFFT = cufftHandle;
    const auto VendorCreate = cufftPlan2d;
    const auto VendorDestroy = cufftDestroy;
#  ifdef AMREX_USE_FLOAT
    const auto VendorExecute = cufftExecC2C;
    const auto cufft_type = CUFFT_C2C;
    using cufftComplex = cuComplex;
#  else
    const auto VendorExecute = cufftExecZ2Z;
    const auto cufft_type = CUFFT_Z2Z;
    using cufftComplex = cuDoubleComplex;
#  endif
#elif defined(AMREX_USE_HIP)
    using VendorFFT = rocfft_plan;
    const auto VendorDestroy = rocfft_plan_destroy;
// TODO
#else
#  ifdef AMREX_USE_FLOAT
    using VendorFFT = fftwf_plan;
    using FFTWComplex = fftwf_complex;
    const auto VendorCreate = fftwf_plan_dft_2d;
    const auto VendorExecute = fftwf_execute;
    const auto VendorDestroy = fftwf_destroy_plan;
#  else
    using VendorFFT = fftw_plan;
    using FFTWComplex = fftw_complex;
    const auto VendorCreate = fftw_plan_dft_2d;
    const auto VendorExecute = fftw_execute;
    const auto VendorDestroy = fftw_destroy_plan;
#  endif
#endif
}

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichLaserSlice {
    // n stands for the time step, j for the longitudinal slice.
    // n00 is time step n, nm1 is n-1 and np1 is n+1. Similar notation for slice j.
    enum slice { nm1j00, nm1jp1, nm1jp2, n00j00, n00jp1, n00jp2, np1j00, np1jp1, np1jp2, N };
};

class Fields;

class Laser
{

    using SpectralFieldLoc = amrex::BaseFab <amrex::GpuComplex<amrex::Real>>;

private:

    static constexpr int m_nslices = WhichLaserSlice::N;

public:

    /** Constructor */
    explicit Laser ()
    {
        ReadParameters();
    }

    ~Laser ()
    {
        if (!m_use_laser) return;
        if (m_solver_type == "fft") {
            LaserFFT::VendorDestroy( m_plan_fwd );
            LaserFFT::VendorDestroy( m_plan_bkw );
        }
    }

    void ReadParameters ();

    /** get function for the 2D slices
     * \param[in] islice slice index
     */
    amrex::MultiFab& getSlices (int islice) {return m_slices[islice]; }

    /** get function for the 2D slices (const version)
     * \param[in] islice slice index
     */
    const amrex::MultiFab& getSlices (int islice) const {return m_slices[islice]; }

    /** Return the 3D FArrayBox containing the laser envelope, const version */
    const amrex::FArrayBox& getFAB () const {return m_F;}

    /** Retun the 3D FArrayBox containing the laser envelope, non-const version */
    amrex::FArrayBox& getFAB () {return m_F;}

    /** getter function, whether 3D envelope is store in host memory */
    int is3dOnHost () const {return m_3d_on_host;}

    /** \brief Allocate laser multifab
     * \param[in] slice_ba box array of the slice
     * \param[in] slice_dm corresponding distribution mapping
     */
    void InitData (const amrex::BoxArray& slice_ba,
                   const amrex::DistributionMapping& slice_dm);

    /** \brief Initialize 3D laser field on current box.
     *
     * \param[in] step time step of the simulation
     * \param[in] bx Box on which the laser field is initialized
     * \param[in] gm Geometry of the problem
     */
    void Init3DEnvelope (int step, amrex::Box bx, const amrex::Geometry& gm);

    /** \brief Copy from 2D slice on device to 3D array on host, and vice-versa
     *
     * \param[in] isl slice index, referring to the 3D slice
     * \param[in] to3d if true, copy from 2D slice to 3D array. Otherwise, the other way.
     */
    void Copy (int isl, bool to3d);

    /** Wrapper function to advance a laser slice by 1 time step.
     * \param[in] fields Field object
     * \param[in] geom Geometry object
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     */
    void AdvanceSlice (const Fields& fields, const amrex::Geometry& geom, amrex::Real dt, int step);

    /** Advance a laser slice by 1 time step using a multigrid solver.
     * The complex phase of the envelope is evaluated on-axis only, but can be generalized to everywhere.
     *
     * \param[in] fields Field object
     * \param[in] geom Geometry object
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     */
    void AdvanceSliceMG (const Fields& fields, const amrex::Geometry& geom, amrex::Real dt, int step);

    /** Advance a laser slice by 1 time step using a FFT solver.
     * The complex phase of the envelope is evaluated on-axis only.
     *
     * \param[in] fields Field object
     * \param[in] geom Geometry object
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     */
    void AdvanceSliceFFT (const Fields& fields, const amrex::Geometry& geom, amrex::Real dt, int step);

    /** Initialize 1 longitudinal slice of the laser, and store it in n00j00 (current time step)
     * and nm1j00 (previous time step).
     *
     * \param[in] geom Geometry object for the slice
     * \param[in] islice slice index
     */
    void InitLaserSlice (const amrex::Geometry& geom, const int islice);

    bool m_use_laser {false}; /**< whether a laser is used or not */

private:

    std::string m_name = "laser"; /**< name of the laser */
    amrex::Real m_a0 {0.}; /**< Laser peak normalized amplitude */
    amrex::RealVect m_w0 {0., 0., 0.}; /**< Laser waist in x and y (the third value is omitted) */
    amrex::Real m_L0 {0.}; /**< Laser length (HW 1/e in amplitude) */
    amrex::Real m_tau {0.}; /**< Laser duration (HW 1/e in amplitude) */
    amrex::Real m_lambda0 {0.}; /**< Laser central wavelength */
    /** Average position of the Gaussian laser pulse */
    amrex::RealVect m_position_mean {0., 0., 0.};
    int m_3d_on_host {0};/** Whether the 3D laser envelope is stored in host or device memory */
    amrex::Real m_focal_distance {0.};
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards = {-1, -1, -1};
    std::string m_solver_type = "multigrid";
    bool m_use_phase {true};
    amrex::Box m_slice_box;

    /** Nb fields in 3D array: new_real, new_imag, old_real, old_imag */
    int m_nfields_3d {4};
    /** 3D laser data. Could be a vector over levels if MR needed for laser */
    amrex::FArrayBox m_F;
    /** Array of N slices required to compute current slice */
    std::array<amrex::MultiFab, m_nslices> m_slices;
    amrex::Real m_MG_tolerance_rel = 1.e-4;
    amrex::Real m_MG_tolerance_abs = 0.;
    int m_MG_verbose = 0;
    /** Whether to use time-averaged RHS in envelope solver. */
    bool m_MG_average_rhs = true;
    /** hpmg solver for the envelope solver */
    std::unique_ptr<hpmg::MultiGrid> m_mg;

    // Elements for the FFT-based laser envelope solver
    // This could belong to AnyFFT etc., with 2 caveats:
    // - This solver operates on a FArrayBox instead of a MultiFab, which is more adequate
    // - The array in position space must be Complex rather than real, which takes up quite some
    //   rewriting, see https://github.com/MaxThevenet/hipace/tree/laser_solve,
    // not sure what the best way to proceed is.
    /** FFTW plan for forward C2C transform to solve Complex Poisson equation */
    LaserFFT::VendorFFT m_plan_fwd;
    /** FFTW plan for backward C2C transform to solve Complex Poisson equation */
    LaserFFT::VendorFFT m_plan_bkw;
    /** Complex FAB to store the solution (e.g. laser envelope on current slice) */
    SpectralFieldLoc m_sol;
    /** Complex FAB to store the RHS in position space */
    SpectralFieldLoc m_rhs;
    /** Complex FAB to store the RHS in Fourier space */
    SpectralFieldLoc m_rhs_fourier;
#ifdef AMREX_USE_CUDA
    cufftResult m_result_fwd;
    cufftResult m_result_bkw;
#endif
};

#endif // LASER_H_
