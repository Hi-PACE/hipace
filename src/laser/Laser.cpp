#include "Laser.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#ifdef AMREX_USE_CUDA
#  include "fields/fft_poisson_solver/fft/CuFFTUtils.H"
#endif
#include <AMReX_GpuComplex.H>

#ifdef AMREX_USE_CUDA
#  include <cufft.h>
#elif defined(AMREX_USE_HIP)
#  include <cstddef>
#  include <rocfft.h>
#else
#  include <fftw3.h>
#endif

void
Laser::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    m_use_laser = queryWithParser(pp, "a0", m_a0);
    if (!m_use_laser) return;
#if defined(AMREX_USE_HIP)
    amrex::Abort("Laser solver not implemented with HIP");
#endif
    amrex::Vector<amrex::Real> tmp_vector;
    if (queryWithParser(pp, "w0", tmp_vector)){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(tmp_vector.size() == 2,
        "The laser waist w0 must be provided in x and y, "
        "so laser.w0 should contain 2 values");
        for (int i=0; i<2; i++) m_w0[i] = tmp_vector[i];
    }

    bool length_is_specified = queryWithParser(pp, "L0", m_L0);;
    bool duration_is_specified = queryWithParser(pp, "tau", m_tau);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( length_is_specified + duration_is_specified == 1,
        "Please specify exlusively either the pulse length L0 or the duration tau of the laser");
    if (duration_is_specified) m_L0 = m_tau/get_phys_const().c;
    getWithParser(pp, "lambda0", m_lambda0);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    queryWithParser(pp, "focal_distance", m_focal_distance);
    queryWithParser(pp, "position_mean", loc_array);
    queryWithParser(pp, "3d_on_host", m_3d_on_host);
    queryWithParser(pp, "solver_type", m_solver_type);
    AMREX_ALWAYS_ASSERT(m_solver_type == "explicit" || m_solver_type == "fft");
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
}


void
Laser::InitData (const amrex::BoxArray& slice_ba,
                 const amrex::DistributionMapping& slice_dm)
{
    if (!m_use_laser) return;

    HIPACE_PROFILE("Laser::InitData()");

    // Alloc 2D slices
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = std::max(1, Hipace::m_depos_order_xy);
    m_slices_nguards = {nguards_xy, nguards_xy, 0};
    AMREX_ALWAYS_ASSERT(WhichLaserSlice::N == m_nslices);
    for (int islice=0; islice<WhichLaserSlice::N; islice++) {
        // 2 components for complex numbers.
        m_slices[islice].define(
            slice_ba, slice_dm, 2, m_slices_nguards,
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[islice].setVal(0.0);
    }

    const amrex::Box& bx = slice_ba[0];
    m_sol.resize(bx, 1, amrex::The_Arena());
    m_rhs.resize(bx, 1, amrex::The_Arena());
    m_rhs_fourier.resize(bx, 1, amrex::The_Arena());

    if (m_solver_type == "fft") {

        // Create FFT plans
        amrex::IntVect fft_size = bx.length();

#ifdef AMREX_USE_CUDA
        cufftResult result;
        // Forward FFT plan
        result = LaserFFT::VendorCreate(
            &(m_plan_fwd), fft_size[1], fft_size[0], CUFFT_Z2Z);
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " cufftplan failed! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
        // Backward FFT plan
        result = LaserFFT::VendorCreate(
            &(m_plan_bkw), fft_size[1], fft_size[0], CUFFT_Z2Z);
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " cufftplan failed! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
#elif defined(AMREX_USE_HIP)
#else
        // Forward FFT plan
        m_plan_fwd = LaserFFT::VendorCreate(
            fft_size[1], fft_size[0],
            reinterpret_cast<fftw_complex*>(m_rhs.dataPtr()),
            reinterpret_cast<fftw_complex*>(m_rhs_fourier.dataPtr()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        // Backward FFT plan
        m_plan_bkw = LaserFFT::VendorCreate(
            fft_size[1], fft_size[0],
            reinterpret_cast<fftw_complex*>(m_rhs_fourier.dataPtr()),
            reinterpret_cast<fftw_complex*>(m_sol.dataPtr()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
    }
}

void
Laser::Init3DEnvelope (int step, amrex::Box bx, const amrex::Geometry& gm, const amrex::Real dt)
{

    if (!m_use_laser) return;

    HIPACE_PROFILE("Laser::Init3DEnvelope()");
    // Allocate the 3D field on this box
    // Note: box has no guard cells
    m_F.resize(bx, m_nfields_3d, m_3d_on_host ? amrex::The_Pinned_Arena() : amrex::The_Arena());

    if (step > 0) return;

    // Loop over slices
    for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
        // Compute initial field on the current (device) slice
        // n00j00 for current, nm1j00 for old
        InitLaserSlice(gm, isl, dt);
        // Copy: (device) slice to (host) 3D array
        // A = np1j00, Aold = n00j00
        Copy(isl, true, true);
    }
}

void
Laser::Copy (int isl, bool to3d, bool init)
{
    if (!m_use_laser) return;

    using namespace amrex::literals;

    HIPACE_PROFILE("Laser::Copy()");

    amrex::MultiFab& nm1j00 = m_slices[WhichLaserSlice::nm1j00];
    amrex::MultiFab& nm1jp1 = m_slices[WhichLaserSlice::nm1jp1];
    amrex::MultiFab& nm1jp2 = m_slices[WhichLaserSlice::nm1jp2];
    amrex::MultiFab& n00j00 = m_slices[WhichLaserSlice::n00j00];
    amrex::MultiFab& n00jp1 = m_slices[WhichLaserSlice::n00jp1];
    amrex::MultiFab& n00jp2 = m_slices[WhichLaserSlice::n00jp2];
    amrex::MultiFab& np1j00 = m_slices[WhichLaserSlice::np1j00];
    amrex::MultiFab& np1jp1 = m_slices[WhichLaserSlice::np1jp1];
    amrex::MultiFab& np1jp2 = m_slices[WhichLaserSlice::np1jp2];
    const int izmax = m_F.box().bigEnd(2);
    const int izmin = m_F.box().smallEnd(2);

    for ( amrex::MFIter mfi(n00j00, false); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        Array3<amrex::Real> nm1j00_arr = nm1j00.array(mfi);
        Array3<amrex::Real> nm1jp1_arr = nm1jp1.array(mfi);
        Array3<amrex::Real> nm1jp2_arr = nm1jp2.array(mfi);
        Array3<amrex::Real> n00j00_arr = n00j00.array(mfi);
        Array3<amrex::Real> n00jp1_arr = n00jp1.array(mfi);
        Array3<amrex::Real> n00jp2_arr = n00jp2.array(mfi);
        Array3<amrex::Real> np1j00_arr = np1j00.array(mfi);
        Array3<amrex::Real> np1jp1_arr = np1jp1.array(mfi);
        Array3<amrex::Real> np1jp2_arr = np1jp2.array(mfi);
        amrex::Array4<amrex::Real> host_arr = m_F.array();
        amrex::ParallelFor(
        bx, 2,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
        {
            // +2 in 3D array means old.
            // 2 components for complex numbers.
            if (to3d){
                // next time slice into new host
                host_arr(i,j,isl,n) = np1j00_arr(i,j,n);
                if (init) {
                    host_arr(i,j,isl,n+2) = n00j00_arr(i,j,n);
                }
                // this slice into old host
                // Cannot update slice isl or isl+1 of old array because we'll need the
                // previous value when computing the next slice.
                if (!init) {
                    if (isl+2 <= izmax){
                        host_arr(i,j,isl+2,n+2) = n00jp2_arr(i,j,n);
                    }
                }
            } else {
                nm1j00_arr(i,j,n) = host_arr(i,j,isl,n+2);
                nm1jp1_arr(i,j,n) = isl+1 <= izmax ? host_arr(i,j,isl+1,n+2) : 0._rt;
                nm1jp2_arr(i,j,n) = isl+2 <= izmax ? host_arr(i,j,isl+2,n+2) : 0._rt;
                n00jp2_arr(i,j,n) = n00jp1_arr(i,j,n);
                n00jp1_arr(i,j,n) = n00j00_arr(i,j,n);
                n00j00_arr(i,j,n) = host_arr(i,j,isl,n);
                np1jp2_arr(i,j,n) = np1jp1_arr(i,j,n);
                np1jp1_arr(i,j,n) = np1j00_arr(i,j,n);
            }
        });
    }
}

void
Laser::AdvanceSlice (const Fields& fields, const amrex::Geometry& geom, const amrex::Real dt)
{

    if (!m_use_laser) return;

    if (m_solver_type == "explicit") {
        AdvanceSliceMG(fields, geom, dt);
    } else if (m_solver_type == "fft") {
        AdvanceSliceFFT(fields, geom, dt);
    } else {
        amrex::Abort("<laser name>.solver_type must be fft or explicit");
    }
}

void
Laser::AdvanceSliceMG (const Fields& fields, const amrex::Geometry& geom, const amrex::Real dt)
{

    HIPACE_PROFILE("Laser::AdvanceSliceMG()");

    using namespace amrex::literals;

    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);
    const amrex::Real dz = geom.CellSize(2);

    const amrex::Real c = get_phys_const().c;
    const amrex::Real k0 = 2.*MathConst::pi/m_lambda0;

    amrex::MultiFab& nm1j00 = m_slices[WhichLaserSlice::nm1j00];
    amrex::MultiFab& nm1jp1 = m_slices[WhichLaserSlice::nm1jp1];
    amrex::MultiFab& nm1jp2 = m_slices[WhichLaserSlice::nm1jp2];
    amrex::MultiFab& n00j00 = m_slices[WhichLaserSlice::n00j00];
    amrex::MultiFab& n00jp1 = m_slices[WhichLaserSlice::n00jp1];
    amrex::MultiFab& n00jp2 = m_slices[WhichLaserSlice::n00jp2];
    amrex::MultiFab& np1j00 = m_slices[WhichLaserSlice::np1j00];
    amrex::MultiFab& np1jp1 = m_slices[WhichLaserSlice::np1jp1];
    amrex::MultiFab& np1jp2 = m_slices[WhichLaserSlice::np1jp2];

    amrex::FArrayBox rhs;
    amrex::FArrayBox acoeff_imag;
    amrex::Real acoeff_real = -3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt);

    for ( amrex::MFIter mfi(n00j00, false); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        const int imin = bx.smallEnd(0);
        const int imax = bx.bigEnd  (0);
        const int jmin = bx.smallEnd(1);
        const int jmax = bx.bigEnd  (1);

        acoeff_imag.resize(bx, 1, amrex::The_Arena());
        rhs.resize(bx, 2, amrex::The_Arena());
        Array3<amrex::Real> nm1j00_arr = nm1j00.array(mfi);
        Array3<amrex::Real> nm1jp1_arr = nm1jp1.array(mfi);
        Array3<amrex::Real> nm1jp2_arr = nm1jp2.array(mfi);
        Array3<amrex::Real> n00j00_arr = n00j00.array(mfi);
        Array3<amrex::Real> n00jp1_arr = n00jp1.array(mfi);
        Array3<amrex::Real> n00jp2_arr = n00jp2.array(mfi);
        Array3<amrex::Real> np1jp1_arr = np1jp1.array(mfi);
        Array3<amrex::Real> np1jp2_arr = np1jp2.array(mfi);
        Array3<amrex::Real> rhs_arr    = rhs.array();
        Array3<amrex::Real> acoeff_imag_arr = acoeff_imag.array();

        amrex::ParallelFor(
            bx, 1,
            [=] AMREX_GPU_DEVICE(int i, int j, int, int) noexcept
            {
                // Calculate complex arguments (theta) needed
                const amrex::Real tj00 = std::atan2(
                    0.25*(n00j00_arr(i-1,j-1,1) + n00j00_arr(i-1,j,1) + n00j00_arr(i,j-1,1) + n00j00_arr(i,j,1)),
                    0.25*(n00j00_arr(i-1,j-1,0) + n00j00_arr(i-1,j,0) + n00j00_arr(i,j-1,0) + n00j00_arr(i,j,0)));
                const amrex::Real tjp1 = std::atan2(
                    0.25*(n00jp1_arr(i-1,j-1,1) + n00jp1_arr(i-1,j,1) + n00jp1_arr(i,j-1,1) + n00jp1_arr(i,j,1)),
                    0.25*(n00jp1_arr(i-1,j-1,0) + n00jp1_arr(i-1,j,0) + n00jp1_arr(i,j-1,0) + n00jp1_arr(i,j,0)));
                const amrex::Real tjp2 = std::atan2(
                    0.25*(n00jp2_arr(i-1,j-1,1) + n00jp2_arr(i-1,j,1) + n00jp2_arr(i,j-1,1) + n00jp2_arr(i,j,1)),
                    0.25*(n00jp2_arr(i-1,j-1,0) + n00jp2_arr(i-1,j,0) + n00jp2_arr(i,j-1,0) + n00jp2_arr(i,j,0)));
                amrex::Real dt1 = tj00 - tjp1;
                amrex::Real dt2 = tj00 - tjp2;
                if (dt1 <-1.5_rt*MathConst::pi) dt1 += 2._rt*MathConst::pi;
                if (dt1 > 1.5_rt*MathConst::pi) dt1 -= 2._rt*MathConst::pi;
                if (dt2 <-1.5_rt*MathConst::pi) dt2 += 2._rt*MathConst::pi;
                if (dt2 > 1.5_rt*MathConst::pi) dt2 -= 2._rt*MathConst::pi;
                amrex::Real cdt1 = std::cos(dt1);
                amrex::Real cdt2 = std::cos(dt2);
                amrex::Real sdt1 = std::sin(dt1);
                amrex::Real sdt2 = std::sin(dt2);
                // Transverse Laplacian of real and imaginary parts of A_j^n-1
                amrex::Real lapR = i>imin && i<imax && j>jmin && j<jmax ?
                    (nm1j00_arr(i+1,j,0)+nm1j00_arr(i-1,j,0)-2._rt*nm1j00_arr(i,j,0))/(dx*dx) +
                    (nm1j00_arr(i,j+1,0)+nm1j00_arr(i,j-1,0)-2._rt*nm1j00_arr(i,j,0))/(dy*dy) : 0._rt;
                amrex::Real lapI = i>imin && i<imax && j>jmin && j<jmax ?
                    (nm1j00_arr(i+1,j,1)+nm1j00_arr(i-1,j,1)-2._rt*nm1j00_arr(i,j,1))/(dx*dx) +
                    (nm1j00_arr(i,j+1,1)+nm1j00_arr(i,j-1,1)-2._rt*nm1j00_arr(i,j,1))/(dy*dy) : 0._rt;
                // D_j^n as defined in Benedetti's 2017 paper
                amrex::Real djn = -( 3._rt*tj00 - 4._rt*tjp1 + tjp2 ) / (2._rt*dz);

                // Imag acoeff term (the Real part is just a scalar defined above)
                acoeff_imag_arr(i,j,0) = -2._rt*( k0 + djn ) / (c*dt);
                // Real RHS term
                rhs_arr(i,j,0) =
                    + 4._rt/(c*dt*dz)*((np1jp1_arr(i,j,0)-nm1jp1_arr(i,j,0))*cdt1 -
                                       (np1jp1_arr(i,j,1)-nm1jp1_arr(i,j,1))*sdt1)
                    - 1._rt/(c*dt*dz)*((np1jp2_arr(i,j,0)-nm1jp2_arr(i,j,0))*cdt2 -
                                       (np1jp2_arr(i,j,1)-nm1jp2_arr(i,j,1))*sdt2)
                    - 4._rt/(c*c*dt*dt)*n00j00_arr(i,j,0)
                    + 0.0000000
                    - lapR
                    + 3._rt/(c*dt*dz)  * nm1j00_arr(i,j,0)
                    - 2._rt/(c*dt)*djn * nm1j00_arr(i,j,1)
                    + 2._rt/(c*c*dt*dt) * nm1j00_arr(i,j,0)
                    - 2._rt*k0/(c*dt)*nm1j00_arr(i,j,1);

                // Imag RHS term
                rhs_arr(i,j,1) =
                    + 4._rt/(c*dt*dz)*((np1jp1_arr(i,j,1)-nm1jp1_arr(i,j,1))*cdt1 +
                                       (np1jp1_arr(i,j,0)-nm1jp1_arr(i,j,0))*sdt1)
                    - 1._rt/(c*dt*dz)*((np1jp2_arr(i,j,1)-nm1jp2_arr(i,j,1))*cdt2 +
                                       (np1jp2_arr(i,j,0)-nm1jp2_arr(i,j,0))*sdt2)
                    - 4._rt/(c*c*dt*dt)*n00j00_arr(i,j,1)
                    - 0.0000000
                    - lapI
                    + 3._rt/(c*dt*dz)  * nm1j00_arr(i,j,1)
                    + 2._rt/(c*dt)*djn * nm1j00_arr(i,j,0)
                    + 2._rt/(c*c*dt*dt) * nm1j00_arr(i,j,1)
                    + 2._rt*k0/(c*dt)*nm1j00_arr(i,j,0);
            });
    }

    // construct slice geometry
    // Set the lo and hi of domain and probdomain in the z direction
    amrex::RealBox tmp_probdom({AMREX_D_DECL(geom.ProbLo(Direction::x),
                                             geom.ProbLo(Direction::y),
                                             geom.ProbLo(Direction::z))},
                               {AMREX_D_DECL(geom.ProbHi(Direction::x),
                                             geom.ProbHi(Direction::y),
                                             geom.ProbHi(Direction::z))});
    amrex::Box tmp_dom = geom.Domain();
    const amrex::Real hi = geom.ProbHi(Direction::z);
    const amrex::Real lo = hi - geom.CellSize(2);
    tmp_probdom.setLo(Direction::z, lo);
    tmp_probdom.setHi(Direction::z, hi);
    tmp_dom.setSmall(Direction::z, 0);
    tmp_dom.setBig(Direction::z, 0);
    amrex::Geometry slice_geom = amrex::Geometry(
        tmp_dom, tmp_probdom, geom.Coord(), geom.isPeriodic());

    slice_geom.setPeriodicity({0,0,0});
    if (!m_mg) {
        m_mg = std::make_unique<hpmg::MultiGrid>(slice_geom);
    }

    const int max_iters = 200;
    m_mg->solve2(np1j00[0], rhs, acoeff_real, acoeff_imag, m_MG_tolerance_rel, m_MG_tolerance_abs,
    max_iters, m_MG_verbose);
}

void
Laser::AdvanceSliceFFT (const Fields& fields, const amrex::Geometry& geom, const amrex::Real dt)
{

    HIPACE_PROFILE("Laser::AdvanceSliceExplicit()");

    using namespace amrex;
    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;
    constexpr Complex I(0.,1.);

    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);
    const amrex::Real dz = geom.CellSize(2);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;
    const amrex::Real k0 = 2.*MathConst::pi/m_lambda0;
    const amrex::Real chi_fac = phc.q_e/(c*c*phc.m_e*phc.ep0);

    amrex::MultiFab& nm1j00 = m_slices[WhichLaserSlice::nm1j00];
    amrex::MultiFab& nm1jp1 = m_slices[WhichLaserSlice::nm1jp1];
    amrex::MultiFab& nm1jp2 = m_slices[WhichLaserSlice::nm1jp2];
    amrex::MultiFab& n00j00 = m_slices[WhichLaserSlice::n00j00];
    amrex::MultiFab& n00jp1 = m_slices[WhichLaserSlice::n00jp1];
    amrex::MultiFab& n00jp2 = m_slices[WhichLaserSlice::n00jp2];
    amrex::MultiFab& np1j00 = m_slices[WhichLaserSlice::np1j00];
    amrex::MultiFab& np1jp1 = m_slices[WhichLaserSlice::np1jp1];
    amrex::MultiFab& np1jp2 = m_slices[WhichLaserSlice::np1jp2];

    for ( amrex::MFIter mfi(n00j00, false); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        const int imin = bx.smallEnd(0);
        const int imax = bx.bigEnd  (0);
        const int jmin = bx.smallEnd(1);
        const int jmax = bx.bigEnd  (1);

        // solution: complex array
        // The right-hand side is computed and stored in rhs
        // Then rhs is Fourier-transformed into rhs_fourier, then multiplied by -1/(k**2+a)
        // rhs_fourier is FFT-back-transformed to sol, and sol is normalized and copied into np1j00.
        Array3<Complex> sol_arr = m_sol.array();
        Array3<Complex> rhs_arr = m_rhs.array();
        Array3<Complex> rhs_fourier_arr = m_rhs_fourier.array();

        Array3<amrex::Real> nm1j00_arr = nm1j00.array(mfi);
        Array3<amrex::Real> nm1jp1_arr = nm1jp1.array(mfi);
        Array3<amrex::Real> nm1jp2_arr = nm1jp2.array(mfi);
        Array3<amrex::Real> n00j00_arr = n00j00.array(mfi);
        Array3<amrex::Real> n00jp1_arr = n00jp1.array(mfi);
        Array3<amrex::Real> n00jp2_arr = n00jp2.array(mfi);
        Array3<amrex::Real> np1j00_arr = np1j00.array(mfi);
        Array3<amrex::Real> np1jp1_arr = np1jp1.array(mfi);
        Array3<amrex::Real> np1jp2_arr = np1jp2.array(mfi);

        constexpr int lev = 0;
        const amrex::FArrayBox& isl_fab = fields.getSlices(lev, WhichSlice::This)[mfi];
        Array3<amrex::Real const> const isl_arr = isl_fab.array();
        const int chi = Comps[WhichSlice::This]["chi"];

        int const Nx = bx.length(0);
        int const Ny = bx.length(1);

        // Get the central point. Useful to get the on-axis phase and calculate kx and ky
        int const imid = (Nx+1)/2;
        int const jmid = (Ny+1)/2;

        // Calculate complex arguments (theta) needed
        // Just once, on axis, as done in Wake-T
        // This is done with a reduce operation, returning the sum of the four elements nearest
        // the axis (both real and imag parts, and for the 3 arrays relevant) ...
        ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum,
                  ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
        ReduceData<Real, Real, Real, Real, Real, Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int) -> ReduceTuple
        {
            if ( ( i == imid-1 || i == imid ) && ( j == jmid-1 || j == jmid ) ) {
                return {
                    n00j00_arr(i,j,0), n00j00_arr(i,j,1),
                    n00jp1_arr(i,j,0), n00jp1_arr(i,j,1),
                    n00jp2_arr(i,j,0), n00jp2_arr(i,j,1)
                };
            } else {
                return {0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt};
            }
        });
        // ... and taking the argument of the resulting complex number.
        ReduceTuple hv = reduce_data.value(reduce_op);
        const amrex::Real tj00 = std::atan2(amrex::get<1>(hv), amrex::get<0>(hv));
        const amrex::Real tjp1 = std::atan2(amrex::get<3>(hv), amrex::get<2>(hv));
        const amrex::Real tjp2 = std::atan2(amrex::get<5>(hv), amrex::get<4>(hv));

        amrex::Real dt1 = tj00 - tjp1;
        amrex::Real dt2 = tj00 - tjp2;
        if (dt1 <-1.5_rt*MathConst::pi) dt1 += 2._rt*MathConst::pi;
        if (dt1 > 1.5_rt*MathConst::pi) dt1 -= 2._rt*MathConst::pi;
        if (dt2 <-1.5_rt*MathConst::pi) dt2 += 2._rt*MathConst::pi;
        if (dt2 > 1.5_rt*MathConst::pi) dt2 -= 2._rt*MathConst::pi;
        Complex edt1 = amrex::exp(I*dt1);
        Complex edt2 = amrex::exp(I*dt2);

        // D_j^n as defined in Benedetti's 2017 paper
        amrex::Real djn = ( -3._rt*tj00 + 4._rt*tjp1 - tjp2 ) / (2._rt*dz);
        amrex::ParallelFor(
            bx, 1,
            [=] AMREX_GPU_DEVICE(int i, int j, int, int n) noexcept
            {
                // Transverse Laplacian of real and imaginary parts of A_j^n-1
                amrex::Real lapR = i>imin && i<imax && j>jmin && j<jmax ?
                    (nm1j00_arr(i+1,j,0)+nm1j00_arr(i-1,j,0)-2._rt*nm1j00_arr(i,j,0))/(dx*dx) +
                    (nm1j00_arr(i,j+1,0)+nm1j00_arr(i,j-1,0)-2._rt*nm1j00_arr(i,j,0))/(dy*dy) : 0._rt;
                amrex::Real lapI = i>imin && i<imax && j>jmin && j<jmax ?
                    (nm1j00_arr(i+1,j,1)+nm1j00_arr(i-1,j,1)-2._rt*nm1j00_arr(i,j,1))/(dx*dx) +
                    (nm1j00_arr(i,j+1,1)+nm1j00_arr(i,j-1,1)-2._rt*nm1j00_arr(i,j,1))/(dy*dy) : 0._rt;
                const Complex anp1jp1 = np1jp1_arr(i,j,0) + I * np1jp1_arr(i,j,1);
                const Complex anm1jp1 = nm1jp1_arr(i,j,0) + I * nm1jp1_arr(i,j,1);
                const Complex anp1jp2 = np1jp2_arr(i,j,0) + I * np1jp2_arr(i,j,1);
                const Complex anm1jp2 = nm1jp2_arr(i,j,0) + I * nm1jp2_arr(i,j,1);
                const Complex an00j00 = n00j00_arr(i,j,0) + I * n00j00_arr(i,j,1);
                const Complex anm1j00 = nm1j00_arr(i,j,0) + I * nm1j00_arr(i,j,1);
                const Complex lapA = lapR + I*lapI;
                const Complex rhs =
                    + 4._rt/(c*dt*dz)*(-anp1jp1+anm1jp1)*edt1
                    + 1._rt/(c*dt*dz)*(+anp1jp2-anm1jp2)*edt2
                    - 4._rt/(c*c*dt*dt)*an00j00
// Not sure whether this is + or -, I believe - and this is consistent with
// the physical effect of plasma on a laser, but double checking would be good.
//                    + 2._rt * chi_fac * isl_arr(i,j,chi) * an00j00
                    - 2._rt * chi_fac * isl_arr(i,j,chi) * an00j00
                    - lapA
                    + ( -3._rt/(c*dt*dz) + 2._rt*I*djn/(c*dt) + 2._rt/(c*c*dt*dt) + I*2._rt*k0/(c*dt) ) * anm1j00;
                rhs_arr(i,j,0) = rhs;
            });

        // Transform rhs to Fourier space
#ifdef AMREX_USE_CUDA
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream ( m_plan_fwd, stream);
        cufftResult result;
        result = LaserFFT::VendorExecute(
            m_plan_fwd,
            reinterpret_cast<LaserFFT::cufftComplex*>( m_rhs.dataPtr() ),
            reinterpret_cast<LaserFFT::cufftComplex*>( m_rhs_fourier.dataPtr() ),
            CUFFT_FORWARD);
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
#elif defined(AMREX_USE_HIP)
        amrex::Abort("Not implemented");
#else
        LaserFFT::VendorExecute( m_plan_fwd );
#endif

        // Multiply by appropriate factors in Fourier space
        amrex::Real dkx = 2.*MathConst::pi/geom.ProbLength(0);
        amrex::Real dky = 2.*MathConst::pi/geom.ProbLength(1);
        // acoeff_imag is supposed to be a nx*ny array.
        // For the sake of simplicity, we evaluate it on-axis only.
        const Complex acoeff =
            ( 3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt) )
            - I * 2._rt * ( k0 + djn ) / (c*dt);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // divide rhs_fourier by -(k^2+a)
                amrex::Real kx = (i<imid) ? dkx*i : dkx*(i-Nx);
                amrex::Real ky = (j<jmid) ? dky*j : dky*(j-Ny);
                const Complex inv_k2a = abs(kx*kx + ky*ky + acoeff) > 0. ?
                    1._rt/(kx*kx + ky*ky + acoeff) : 0.;
                rhs_fourier_arr(i,j,k) *= -inv_k2a;
            });

        // Transform rhs to Fourier space to get solution in sol
#ifdef AMREX_USE_CUDA
        cudaStream_t stream_bkw = amrex::Gpu::Device::cudaStream();
        cufftSetStream ( m_plan_bkw, stream_bkw);
        result = LaserFFT::VendorExecute(
            m_plan_bkw,
            reinterpret_cast<LaserFFT::cufftComplex*>( m_rhs_fourier.dataPtr() ),
            reinterpret_cast<LaserFFT::cufftComplex*>( m_sol.dataPtr() ),
            CUFFT_INVERSE);
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
#elif defined(AMREX_USE_HIP)
        amrex::Abort("Not implemented");
#else
        LaserFFT::VendorExecute( m_plan_bkw );
#endif

        // Normalize and store solution in np1j00[0]. Guard cells are filled with 0s.
        amrex::Box grown_bx = bx;
        grown_bx.grow(m_slices_nguards);
        const amrex::Real inv_numPts = 1./bx.numPts();
        amrex::ParallelFor(
            grown_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                if (i>=imin && i<=imax && j>=jmin && j<=jmax) {
                    np1j00_arr(i,j,0) = sol_arr(i,j,0).real() * inv_numPts;
                    np1j00_arr(i,j,1) = sol_arr(i,j,0).imag() * inv_numPts;
                } else {
                    np1j00_arr(i,j,0) = 0._rt;
                    np1j00_arr(i,j,1) = 0._rt;
                }
            });
    }
}

void
Laser::InitLaserSlice (const amrex::Geometry& geom, const int islice, amrex::Real dt)
{
    if (!m_use_laser) return;

    HIPACE_PROFILE("Laser::InitLaserSlice()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;

    // Basic laser parameters and constants
    Complex I(0,1);
    const amrex::Real c = get_phys_const().c;
    const int dcomp = 0; // NOTE, this may change when we use slices with Comps
    const amrex::Real a0 = m_a0;
    const amrex::Real k0 = 2._rt*MathConst::pi/m_lambda0;
    const amrex::Real w0 = m_w0[0];
    const amrex::Real x0 = m_position_mean[0];
    const amrex::Real y0 = m_position_mean[1];
    const amrex::Real z0 = m_position_mean[2];
    const amrex::Real L0 = m_L0;
    const amrex::Real zfoc = m_focal_distance;

    AMREX_ALWAYS_ASSERT(m_w0[0] == m_w0[1]);
    AMREX_ALWAYS_ASSERT(x0 == 0._rt);
    AMREX_ALWAYS_ASSERT(y0 == 0._rt);

    // Get grid properties
    const auto plo = geom.ProbLoArray();
    amrex::Real const * const dx = geom.CellSize();
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    // Envelope quantities common for all this slice
    amrex::MultiFab& np1j00 = getSlices(WhichLaserSlice::np1j00);
    amrex::MultiFab& n00j00 = getSlices(WhichLaserSlice::n00j00);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(np1j00, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const & np1j00_arr = np1j00.array(mfi);
        amrex::Array4<amrex::Real> const & n00j00_arr = n00j00.array(mfi);

        // Initialize a Gaussian laser envelope on slice islice
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real z = plo[2] + (islice+0.5_rt)*dx_arr[2] - zfoc;
                const amrex::Real x = (i+0.5_rt)*dx_arr[0]+plo[0];
                const amrex::Real y = (j+0.5_rt)*dx_arr[1]+plo[1];

                // Compute envelope for time step 0
                Complex diffract_factor = 1._rt + I * z * 2._rt/( k0 * w0 * w0 );
                Complex inv_complex_waist_2 = 1._rt /( w0 * w0 * diffract_factor );
                Complex prefactor = a0/diffract_factor;
                Complex time_exponent = (z-z0+zfoc)*(z-z0+zfoc)/(L0*L0);
                Complex stcfactor = prefactor * amrex::exp( - time_exponent );
                Complex exp_argument = - ( x*x + y*y ) * inv_complex_waist_2;
                Complex envelope = stcfactor * amrex::exp( exp_argument );
                np1j00_arr(i,j,k,dcomp  ) = envelope.real();
                np1j00_arr(i,j,k,dcomp+1) = envelope.imag();

                // Same thing for time step -1
                z -= c * dt;
                diffract_factor = 1._rt + I * z * 2._rt/( k0 * w0 * w0 );
                inv_complex_waist_2 = 1._rt /( w0 * w0 * diffract_factor );
                prefactor = a0/diffract_factor;
                time_exponent = (z-z0+c*dt+zfoc)*(z-z0+c*dt+zfoc)/(L0*L0);
                stcfactor = prefactor * amrex::exp( - time_exponent );
                exp_argument = - ( x*x + y*y ) * inv_complex_waist_2;
                envelope = stcfactor * amrex::exp( exp_argument );
                n00j00_arr(i,j,k,dcomp  ) = envelope.real();
                n00j00_arr(i,j,k,dcomp+1) = envelope.imag();
            }
            );
    }
}
