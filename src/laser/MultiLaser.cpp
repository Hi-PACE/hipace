/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#include "MultiLaser.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/DeprecatedInput.H"
#ifdef AMREX_USE_CUDA
#  include "fields/fft_poisson_solver/fft/CuFFTUtils.H"
#endif
#include "particles/particles_utils/ShapeFactors.H"

#include <AMReX_GpuComplex.H>

#ifdef AMREX_USE_CUDA
#  include <cufft.h>
#elif defined(AMREX_USE_HIP)
#  if __has_include(<rocfft/rocfft.h>)  // ROCm 5.3+
#    include <rocfft/rocfft.h>
#  else
#    include <rocfft.h>
#  endif
#else
#  include <fftw3.h>
#endif

void
MultiLaser::ReadParameters ()
{
    amrex::ParmParse pp("lasers");
    queryWithParser(pp, "names", m_names);

    m_use_laser = m_names[0] != "no_laser";

    if (!m_use_laser) return;
#if defined(AMREX_USE_HIP)
    amrex::Abort("Laser solver not implemented with HIP");
#endif

    m_laser_from_file = queryWithParser(pp, "input_file", m_input_file_path);

    m_nlasers = m_names.size();
    for (int i = 0; i < m_nlasers; ++i) {
        m_all_lasers.emplace_back(Laser(m_names[i], m_laser_from_file));
    }

    if (!m_laser_from_file) {
        getWithParser(pp, "lambda0", m_lambda0);
    }
    DeprecatedInput("lasers", "3d_on_host", "hipace.comms_buffer_on_gpu", "", true);
    queryWithParser(pp, "use_phase", m_use_phase);
    queryWithParser(pp, "solver_type", m_solver_type);
    AMREX_ALWAYS_ASSERT(m_solver_type == "multigrid" || m_solver_type == "fft");

    bool mg_param_given = queryWithParser(pp, "MG_tolerance_rel", m_MG_tolerance_rel);
    mg_param_given += queryWithParser(pp, "MG_tolerance_abs", m_MG_tolerance_abs);
    mg_param_given += queryWithParser(pp, "MG_verbose", m_MG_verbose);
    mg_param_given += queryWithParser(pp, "MG_average_rhs", m_MG_average_rhs);

    // Raise warning if user specifies MG parameters without using the MG solver
    if (mg_param_given && (m_solver_type != "multigrid")) {
        amrex::Print()<<"WARNING: parameters laser.MG_... only active if laser.solver_type = multigrid\n";
    }

    if (m_laser_from_file) {
        queryWithParser(pp, "openPMD_laser_name", m_file_envelope_name);
        queryWithParser(pp, "iteration", m_file_num_iteration);
    }
}


void
MultiLaser::InitData (const amrex::BoxArray& slice_ba,
                      const amrex::DistributionMapping& slice_dm,
                      const amrex::Geometry& geom_3D)
{
    if (!m_use_laser) return;

    HIPACE_PROFILE("MultiLaser::InitData()");

    // Alloc 2D slices
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = (Hipace::m_depos_order_xy + 1) / 2 + 1;
    m_slices_nguards = {nguards_xy, nguards_xy, 0};
    m_slices.define(
        slice_ba, slice_dm, WhichLaserSlice::N, m_slices_nguards,
        amrex::MFInfo().SetArena(amrex::The_Arena()));
    m_slices.setVal(0.0);

    m_laser_geom_3D = geom_3D;
    m_slice_box = slice_ba[0];
    m_sol.resize(m_slice_box, 1, amrex::The_Arena());
    m_rhs.resize(m_slice_box, 1, amrex::The_Arena());
    m_rhs_fourier.resize(m_slice_box, 1, amrex::The_Arena());

    if (m_solver_type == "fft") {

        // Create FFT plans
        amrex::IntVect fft_size = m_slice_box.length();

#ifdef AMREX_USE_CUDA
        cufftResult result;
        // Forward FFT plan
        result = LaserFFT::VendorCreate(
            &(m_plan_fwd), fft_size[1], fft_size[0], LaserFFT::cufft_type);
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " cufftplan failed! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
        // Backward FFT plan
        result = LaserFFT::VendorCreate(
            &(m_plan_bkw), fft_size[1], fft_size[0], LaserFFT::cufft_type);
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " cufftplan failed! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
#elif defined(AMREX_USE_HIP)
        amrex::ignore_unused(fft_size); // TODO: fft solver on AMD
#else
        // Forward FFT plan
        m_plan_fwd = LaserFFT::VendorCreate(
            fft_size[1], fft_size[0],
            reinterpret_cast<LaserFFT::FFTWComplex*>(m_rhs.dataPtr()),
            reinterpret_cast<LaserFFT::FFTWComplex*>(m_rhs_fourier.dataPtr()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        // Backward FFT plan
        m_plan_bkw = LaserFFT::VendorCreate(
            fft_size[1], fft_size[0],
            reinterpret_cast<LaserFFT::FFTWComplex*>(m_rhs_fourier.dataPtr()),
            reinterpret_cast<LaserFFT::FFTWComplex*>(m_sol.dataPtr()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
    }

    if (m_laser_from_file) {
        if (Hipace::HeadRank()) {
            m_F_input_file.resize(m_laser_geom_3D.Domain(), 2, amrex::The_Pinned_Arena());
            GetEnvelopeFromFileHelper(m_laser_geom_3D);
        }
#ifdef AMREX_USE_MPI
        // need to communicate m_lambda0 as it is read in from the input file only by the head rank
        MPI_Bcast(&m_lambda0,
            1,
            amrex::ParallelDescriptor::Mpi_typemap<decltype(m_lambda0)>::type(),
            amrex::ParallelDescriptor::NProcs() - 1, // HeadRank
            amrex::ParallelDescriptor::Communicator());
#endif
    }
}

void
MultiLaser::InitSliceEnvelope (const int islice, const int comp)
{
    if (!m_use_laser) return;

    HIPACE_PROFILE("MultiLaser::InitSliceEnvelope()");

    if (m_laser_from_file) {
        amrex::Box src_box = m_slice_box;
        src_box.setSmall(2, islice);
        src_box.setBig(2, islice);
        m_slices[0].copy<amrex::RunOn::Device>(m_F_input_file, src_box, 0, m_slice_box, comp, 2);
    } else {
        // Compute initial field on the current (device) slice comp and comp + 1
        InitLaserSlice(m_laser_geom_3D, islice, comp);
    }

}

void
MultiLaser::GetEnvelopeFromFileHelper (const amrex::Geometry& gm) {

    HIPACE_PROFILE("MultiLaser::GetEnvelopeFromFileHelper()");
#ifdef HIPACE_USE_OPENPMD
    openPMD::Datatype input_type = openPMD::Datatype::INT;
    {
        // Check what kind of Datatype is used in the Laser file
        auto series = openPMD::Series( m_input_file_path , openPMD::Access::READ_ONLY );

        if(!series.iterations.contains(m_file_num_iteration)) {
            amrex::Abort("Could not find iteration " + std::to_string(m_file_num_iteration) +
                         " in file " + m_input_file_path + "\n");
        }

        auto iteration = series.iterations[m_file_num_iteration];

        if(!iteration.meshes.contains(m_file_envelope_name)) {
            amrex::Abort("Could not find mesh '" + m_file_envelope_name + "' in file "
                + m_input_file_path + "\n");
        }

        auto mesh = iteration.meshes[m_file_envelope_name];

        if (!mesh.containsAttribute("angularFrequency")) {
            amrex::Abort("Could not find Attribute 'angularFrequency' of iteration "
                + std::to_string(m_file_num_iteration) + " in file "
                + m_input_file_path + "\n");
        }

        m_lambda0 = 2.*MathConst::pi*PhysConstSI::c
            / mesh.getAttribute("angularFrequency").get<double>();

        if(!mesh.contains(openPMD::RecordComponent::SCALAR)) {
            amrex::Abort("Could not find component '" +
                std::string(openPMD::RecordComponent::SCALAR) +
                "' in file " + m_input_file_path + "\n");
        }

        input_type = mesh[openPMD::RecordComponent::SCALAR].getDatatype();
    }

    if (input_type == openPMD::Datatype::CFLOAT) {
        GetEnvelopeFromFile<std::complex<float>>(gm);
    } else if (input_type == openPMD::Datatype::CDOUBLE) {
        GetEnvelopeFromFile<std::complex<double>>(gm);
    } else {
        amrex::Abort("Unknown Datatype used in Laser input file. Must use CDOUBLE or CFLOAT\n");
    }
#else
    amrex::ignore_unused(gm);
    amrex::Abort("loading a laser envelope from an external file requires openPMD support: "
                 "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif // HIPACE_USE_OPENPMD
}

template<typename input_type>
void
MultiLaser::GetEnvelopeFromFile (const amrex::Geometry& gm) {

    using namespace amrex::literals;

    HIPACE_PROFILE("MultiLaser::GetEnvelopeFromFile()");
#ifdef HIPACE_USE_OPENPMD
    const PhysConst phc = get_phys_const();
    const amrex::Real clight = phc.c;

    const amrex::Box& domain = gm.Domain();

    auto series = openPMD::Series( m_input_file_path , openPMD::Access::READ_ONLY );
    auto laser = series.iterations[m_file_num_iteration].meshes[m_file_envelope_name];
    auto laser_comp = laser[openPMD::RecordComponent::SCALAR];

    const std::vector<std::string> axis_labels = laser.axisLabels();
    if (axis_labels[0] == "t" && axis_labels[1] == "y" && axis_labels[2] == "x") {
        m_file_geometry = "xyt";
    } else if (axis_labels[0] == "t" && axis_labels[1] == "r") {
        m_file_geometry = "rt";
    } else {
        amrex::Abort("Incorrect axis labels in laser file, must be either t, y, x or t, r");
    }

    const std::shared_ptr<input_type> data = laser_comp.loadChunk<input_type>();
    auto extent = laser_comp.getExtent();
    double unitSI = laser_comp.unitSI();

    // Extract grid offset and grid spacing from laser file
    std::vector<double> offset = laser.gridGlobalOffset();
    std::vector<double> position = laser_comp.position<double>();
    std::vector<double> spacing = laser.gridSpacing<double>();

    //lasy: tyx in C order, tr in C order
    amrex::Dim3 arr_begin = {0, 0, 0};
    amrex::Dim3 arr_end = {static_cast<int>(extent[2]), static_cast<int>(extent[1]),
                            static_cast<int>(extent[0])};
    amrex::Array4<input_type> input_file_arr(data.get(), arr_begin, arr_end, 1);

    //hipace: xyt in Fortran order
    amrex::Array4<amrex::Real> laser_arr = m_F_input_file.array();

    series.flush();

    constexpr int interp_order_xy = 1;
    const amrex::Real dx = gm.CellSize(Direction::x);
    const amrex::Real dy = gm.CellSize(Direction::y);
    const amrex::Real dz = gm.CellSize(Direction::z);
    const amrex::Real xmin = gm.ProbLo(Direction::x)+dx/2;
    const amrex::Real ymin = gm.ProbLo(Direction::y)+dy/2;
    const amrex::Real zmin = gm.ProbLo(Direction::z)+dz/2;
    const amrex::Real zmax = gm.ProbHi(Direction::z)-dz/2;
    const int imin = domain.smallEnd(0);
    const int jmin = domain.smallEnd(1);
    const int kmin = domain.smallEnd(2);

    if (m_file_geometry == "xyt") {
        // Calculate the min and max of the grid from laser file
        amrex::Real ymin_laser = offset[1] + position[1]*spacing[1];
        amrex::Real xmin_laser = offset[2] + position[2]*spacing[2];
        AMREX_ALWAYS_ASSERT(position[0] == 0 && position[1] == 0 && position[2] == 0);


        for (int k = kmin; k <= domain.bigEnd(2); ++k) {
            for (int j = jmin; j <= domain.bigEnd(1); ++j) {
                for (int i = imin; i <= domain.bigEnd(0); ++i) {

                    const amrex::Real x = (i-imin)*dx + xmin;
                    const amrex::Real xmid = (x - xmin_laser)/spacing[2];
                    amrex::Real sx_cell[interp_order_xy+1];
                    const int i_cell = compute_shape_factor<interp_order_xy>(sx_cell, xmid);

                    const amrex::Real y = (j-jmin)*dy + ymin;
                    const amrex::Real ymid = (y - ymin_laser)/spacing[1];
                    amrex::Real sy_cell[interp_order_xy+1];
                    const int j_cell = compute_shape_factor<interp_order_xy>(sy_cell, ymid);

                    const amrex::Real z = (k-kmin)*dz + zmin;
                    const amrex::Real tmid = (zmax-z)/clight/spacing[0];
                    amrex::Real st_cell[interp_order_xy+1];
                    const int k_cell = compute_shape_factor<interp_order_xy>(st_cell, tmid);

                    laser_arr(i, j, k, 0) = 0._rt;
                    laser_arr(i, j, k, 1) = 0._rt;
                    for (int it=0; it<=interp_order_xy; it++){
                        for (int iy=0; iy<=interp_order_xy; iy++){
                            for (int ix=0; ix<=interp_order_xy; ix++){
                                if (i_cell+ix >= 0 && i_cell+ix < static_cast<int>(extent[2]) &&
                                    j_cell+iy >= 0 && j_cell+iy < static_cast<int>(extent[1]) &&
                                    k_cell+it >= 0 && k_cell+it < static_cast<int>(extent[0])) {
                                    laser_arr(i, j, k, 0) += sx_cell[ix] * sy_cell[iy] * st_cell[it] *
                                        static_cast<amrex::Real>(
                                            input_file_arr(i_cell+ix, j_cell+iy, k_cell+it).real() * unitSI
                                        );
                                    laser_arr(i, j, k, 1) += sx_cell[ix] * sy_cell[iy] * st_cell[it] *
                                        static_cast<amrex::Real>(
                                            input_file_arr(i_cell+ix, j_cell+iy, k_cell+it).imag() * unitSI
                                        );
                                }
                            }
                        }
                    } // End of 3 loops (1 per dimension) over laser array from file
                }
            }
        } // End of 3 loops (1 per dimension) over laser array from simulation
    } else if (m_file_geometry == "rt") {

        // extent = {nmodes, nt, nr}

        // Calculate the min and max of the grid from laser file
        amrex::Real rmin_laser = offset[1] + position[1]*spacing[1];
        AMREX_ALWAYS_ASSERT(position[0] == 0 && position[1] == 0);

        for (int k = kmin; k <= domain.bigEnd(2); ++k) {
            for (int j = jmin; j <= domain.bigEnd(1); ++j) {
                for (int i = imin; i <= domain.bigEnd(0); ++i) {

                    const amrex::Real x = (i-imin)*dx + xmin;
                    const amrex::Real y = (j-jmin)*dy + ymin;
                    const amrex::Real r = std::sqrt(x*x + y*y);
                    const amrex::Real theta = std::atan2(y, x);
                    const amrex::Real rmid = (r - rmin_laser)/spacing[1];
                    amrex::Real sr_cell[interp_order_xy+1];
                    const int i_cell = compute_shape_factor<interp_order_xy>(sr_cell, rmid);

                    const amrex::Real z = (k-kmin)*dz + zmin;
                    const amrex::Real tmid = (zmax-z)/clight/spacing[0];
                    amrex::Real st_cell[interp_order_xy+1];
                    const int k_cell = compute_shape_factor<interp_order_xy>(st_cell, tmid);

                    laser_arr(i, j, k, 0) = 0._rt;
                    laser_arr(i, j, k, 1) = 0._rt;
                    for (int it=0; it<=interp_order_xy; it++){
                        for (int ir=0; ir<=interp_order_xy; ir++){
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(i_cell+ir >= 0,
                                "Touching a r<0 cell in laser file reader. Is staggering correct?");
                            if (i_cell+ir < static_cast<int>(extent[2]) &&
                                k_cell+it >= 0 && k_cell+it < static_cast<int>(extent[1])) {
                                // mode 0
                                laser_arr(i, j, k, 0) += sr_cell[ir] * st_cell[it] *
                                    static_cast<amrex::Real>(
                                    input_file_arr(i_cell+ir, k_cell+it, 0).real() * unitSI);
                                laser_arr(i, j, k, 1) += sr_cell[ir] * st_cell[it] *
                                    static_cast<amrex::Real>(
                                    input_file_arr(i_cell+ir, k_cell+it, 0).imag() * unitSI);
                                for (int im=1; im<=static_cast<int>(extent[0])/2; im++) {
                                    // cos(m*theta) part of the mode
                                    laser_arr(i, j, k, 0) += sr_cell[ir] * st_cell[it] *
                                        std::cos(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im-1).real() * unitSI);
                                    laser_arr(i, j, k, 1) += sr_cell[ir] * st_cell[it] *
                                        std::cos(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im-1).imag() * unitSI);
                                    // sin(m*theta) part of the mode
                                    laser_arr(i, j, k, 0) += sr_cell[ir] * st_cell[it] *
                                        std::sin(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im).real() * unitSI);
                                    laser_arr(i, j, k, 1) += sr_cell[ir] * st_cell[it] *
                                        std::sin(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im).imag() * unitSI);
                                } // End of loop over modes of laser array from file
                            }
                        }
                    } // End of 2 loops (1 per RT dimension) over laser array from file
                }
            }
        } // End of 3 loops (1 per dimension) over laser array from simulation
    } // End if statement over file laser geometry (rt or xyt)
#else
    amrex::ignore_unused(gm);
    amrex::Abort("loading a laser envelope from an external file requires openPMD support: "
                 "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif // HIPACE_USE_OPENPMD
}

void
MultiLaser::ShiftLaserSlices ()
{
    if (!m_use_laser) return;

    HIPACE_PROFILE("MultiLaser::ShiftLaserSlices()");

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        const amrex::Box bx = mfi.tilebox();
        Array3<amrex::Real> arr = m_slices.array(mfi);
        amrex::ParallelFor(
        bx, 2,
        [=] AMREX_GPU_DEVICE(int i, int j, int, int n) noexcept
        {
            using namespace WhichLaserSlice;
            // 2 components for complex numbers.
            // Shift slices of step n-1
            const amrex::Real tmp_nm1j00 = arr(i, j, nm1jp2_r + n); // nm1j00_r from host
            arr(i, j, nm1jp2_r + n) = arr(i, j, nm1jp1_r + n);
            arr(i, j, nm1jp1_r + n) = arr(i, j, nm1j00_r + n);
            arr(i, j, nm1j00_r + n) = tmp_nm1j00;
            // Shift slices of step n
            const amrex::Real tmp_n00j00 = arr(i, j, n00jp2_r + n); // n00j00_r from host
            arr(i, j, n00jp2_r + n) = arr(i, j, n00jp1_r + n);
            arr(i, j, n00jp1_r + n) = arr(i, j, n00j00_r + n);
            arr(i, j, n00j00_r + n) = tmp_n00j00;
            // Shift slices of step n+1
            arr(i, j, np1jp2_r + n) = arr(i, j, np1jp1_r + n);
            arr(i, j, np1jp1_r + n) = arr(i, j, np1j00_r + n);
            // np1j00_r will be computed by AdvanceSlice
        });
    }
}

void
MultiLaser::AdvanceSlice (const Fields& fields, amrex::Real dt, int step)
{

    if (!m_use_laser) return;

    if (m_solver_type == "multigrid") {
        AdvanceSliceMG(fields, dt, step);
    } else if (m_solver_type == "fft") {
        AdvanceSliceFFT(fields, dt, step);
    } else {
        amrex::Abort("laser.solver_type must be fft or multigrid");
    }
}

void
MultiLaser::AdvanceSliceMG (const Fields& fields, amrex::Real dt, int step)
{

    HIPACE_PROFILE("MultiLaser::AdvanceSliceMG()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;
    constexpr Complex I(0.,1.);

    const amrex::Real dx = m_laser_geom_3D.CellSize(0);
    const amrex::Real dy = m_laser_geom_3D.CellSize(1);
    const amrex::Real dz = m_laser_geom_3D.CellSize(2);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;
    const amrex::Real k0 = 2.*MathConst::pi/m_lambda0;
    const bool do_avg_rhs = m_MG_average_rhs;

    amrex::FArrayBox rhs_mg;
    amrex::FArrayBox acoeff_real;
    amrex::Real acoeff_real_scalar = 0._rt;
    amrex::Real acoeff_imag_scalar = 0._rt;

    amrex::Real djn {0.};

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        const int imin = bx.smallEnd(0);
        const int imax = bx.bigEnd  (0);
        const int jmin = bx.smallEnd(1);
        const int jmax = bx.bigEnd  (1);

        // need one ghost cell for 2^n-1 MG solve
        acoeff_real.resize(mfi.growntilebox(amrex::IntVect{1, 1, 0}), 1, amrex::The_Arena());
        rhs_mg.resize(mfi.growntilebox(amrex::IntVect{1, 1, 0}), 2, amrex::The_Arena());
        Array3<amrex::Real> arr = m_slices.array(mfi);
        Array3<amrex::Real> rhs_mg_arr = rhs_mg.array();
        Array3<amrex::Real> acoeff_real_arr = acoeff_real.array();
        Array3<Complex> rhs_arr = m_rhs.array();

        constexpr int lev = 0;
        const amrex::FArrayBox& isl_fab = fields.getSlices(lev)[mfi];
        Array3<amrex::Real const> const isl_arr = isl_fab.array();
        const int chi = Comps[WhichSlice::This]["chi"];

        // Calculate phase terms. 0 if !m_use_phase
        amrex::Real tj00 = 0.;
        amrex::Real tjp1 = 0.;
        amrex::Real tjp2 = 0.;

        if (m_use_phase) {
            int const Nx = bx.length(0);
            int const Ny = bx.length(1);

            // Get the central point.
            int const imid = (Nx+1)/2;
            int const jmid = (Ny+1)/2;

            // Calculate complex arguments (theta) needed
            // Just once, on axis, as done in Wake-T
            // This is done with a reduce operation, returning the sum of the four elements nearest
            // the axis (both real and imag parts, and for the 3 arrays relevant) ...
            amrex::ReduceOps<
                amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<
                amrex::Real, amrex::Real, amrex::Real,
                amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int) -> ReduceTuple
                {
                    using namespace WhichLaserSlice;
                    // Even number of transverse cells: average 2 cells
                    // Odd number of cells: only keep central one
                    const bool do_keep_x = Nx % 2 == 0 ?
                        i == imid-1 || i == imid : i == imid;
                    const bool do_keep_y = Ny % 2 == 0 ?
                        j == jmid-1 || j == jmid : j == jmid;
                    if ( do_keep_x && do_keep_y ) {
                        return {
                            arr(i, j, n00j00_r), arr(i, j, n00j00_i),
                            arr(i, j, n00jp1_r), arr(i, j, n00jp1_i),
                            arr(i, j, n00jp2_r), arr(i, j, n00jp2_i)
                        };
                    } else {
                        return {0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt};
                    }
                });
            // ... and taking the argument of the resulting complex number.
            ReduceTuple hv = reduce_data.value(reduce_op);
            tj00 = std::atan2(amrex::get<1>(hv), amrex::get<0>(hv));
            tjp1 = std::atan2(amrex::get<3>(hv), amrex::get<2>(hv));
            tjp2 = std::atan2(amrex::get<5>(hv), amrex::get<4>(hv));
        }

        amrex::Real dt1 = tj00 - tjp1;
        amrex::Real dt2 = tjp1 - tjp2;
        if (dt1 <-1.5_rt*MathConst::pi) dt1 += 2._rt*MathConst::pi;
        if (dt1 > 1.5_rt*MathConst::pi) dt1 -= 2._rt*MathConst::pi;
        if (dt2 <-1.5_rt*MathConst::pi) dt2 += 2._rt*MathConst::pi;
        if (dt2 > 1.5_rt*MathConst::pi) dt2 -= 2._rt*MathConst::pi;
        Complex exp1 = amrex::exp(I*(tj00-tjp1));
        Complex exp2 = amrex::exp(I*(tj00-tjp2));

        // D_j^n as defined in Benedetti's 2017 paper
        djn = ( -3._rt*dt1 + dt2 ) / (2._rt*dz);
        acoeff_real_scalar = step == 0 ? 6._rt/(c*dt*dz)
            : 3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt);
        acoeff_imag_scalar = step == 0 ? -4._rt * ( k0 + djn ) / (c*dt)
            : -2._rt * ( k0 + djn ) / (c*dt);

        amrex::ParallelFor(
            bx, 1,
            [=] AMREX_GPU_DEVICE(int i, int j, int, int) noexcept
            {
                using namespace WhichLaserSlice;
                // Transverse Laplacian of real and imaginary parts of A_j^n-1
                amrex::Real lapR, lapI;
                if (step == 0) {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, n00j00_r)+arr(i-1, j, n00j00_r)-2._rt*arr(i, j, n00j00_r))/(dx*dx) +
                        (arr(i, j+1, n00j00_r)+arr(i, j-1, n00j00_r)-2._rt*arr(i, j, n00j00_r))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, n00j00_i)+arr(i-1, j, n00j00_i)-2._rt*arr(i, j, n00j00_i))/(dx*dx) +
                        (arr(i, j+1, n00j00_i)+arr(i, j-1, n00j00_i)-2._rt*arr(i, j, n00j00_i))/(dy*dy) : 0._rt;
                } else {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, nm1j00_r)+arr(i-1, j, nm1j00_r)-2._rt*arr(i, j, nm1j00_r))/(dx*dx) +
                        (arr(i, j+1, nm1j00_r)+arr(i, j-1, nm1j00_r)-2._rt*arr(i, j, nm1j00_r))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, nm1j00_i)+arr(i-1, j, nm1j00_i)-2._rt*arr(i, j, nm1j00_i))/(dx*dx) +
                        (arr(i, j+1, nm1j00_i)+arr(i, j-1, nm1j00_i)-2._rt*arr(i, j, nm1j00_i))/(dy*dy) : 0._rt;
                }
                const Complex lapA = lapR + I*lapI;
                const Complex an00j00 = arr(i, j, n00j00_r) + I * arr(i, j, n00j00_i);
                const Complex anp1jp1 = arr(i, j, np1jp1_r) + I * arr(i, j, np1jp1_i);
                const Complex anp1jp2 = arr(i, j, np1jp2_r) + I * arr(i, j, np1jp2_i);
                acoeff_real_arr(i,j,0) = do_avg_rhs ?
                    acoeff_real_scalar + isl_arr(i,j,chi) : acoeff_real_scalar;

                Complex rhs;
                if (step == 0) {
                    // First time step: non-centered push to go
                    // from step 0 to step 1 without knowing -1.
                    const Complex an00jp1 = arr(i, j, n00jp1_r) + I * arr(i, j, n00jp1_i);
                    const Complex an00jp2 = arr(i, j, n00jp2_r) + I * arr(i, j, n00jp2_i);
                    rhs =
                        + 8._rt/(c*dt*dz)*(-anp1jp1+an00jp1)*exp1
                        + 2._rt/(c*dt*dz)*(+anp1jp2-an00jp2)*exp2
                        - lapA
                        + ( -6._rt/(c*dt*dz) + 4._rt*I*djn/(c*dt) + I*4._rt*k0/(c*dt) ) * an00j00;
                    if (do_avg_rhs) {
                        rhs += isl_arr(i,j,chi) * an00j00;
                    } else {
                        rhs += isl_arr(i,j,chi) * an00j00 * 2._rt;
                    }
                } else {
                    const Complex anm1jp1 = arr(i, j, nm1jp1_r) + I * arr(i, j, nm1jp1_i);
                    const Complex anm1jp2 = arr(i, j, nm1jp2_r) + I * arr(i, j, nm1jp2_i);
                    const Complex anm1j00 = arr(i, j, nm1j00_r) + I * arr(i, j, nm1j00_i);
                    rhs =
                        + 4._rt/(c*dt*dz)*(-anp1jp1+anm1jp1)*exp1
                        + 1._rt/(c*dt*dz)*(+anp1jp2-anm1jp2)*exp2
                        - 4._rt/(c*c*dt*dt)*an00j00
                        - lapA
                        + ( -3._rt/(c*dt*dz) + 2._rt*I*djn/(c*dt) + 2._rt/(c*c*dt*dt) + I*2._rt*k0/(c*dt) ) * anm1j00;
                    if (do_avg_rhs) {
                        rhs += isl_arr(i,j,chi) * anm1j00;
                    } else {
                        rhs += isl_arr(i,j,chi) * an00j00 * 2._rt;
                    }
                }
                rhs_arr(i,j,0) = rhs;
                rhs_mg_arr(i,j,0) = rhs.real();
                rhs_mg_arr(i,j,1) = rhs.imag();
            });
    }

    if (!m_mg) {
        m_mg = std::make_unique<hpmg::MultiGrid>(m_laser_geom_3D.CellSize(0),
                                                 m_laser_geom_3D.CellSize(1),
                                                 m_slices.boxArray()[0]);
    }

    const int max_iters = 200;
    amrex::MultiFab np1j00 (m_slices, amrex::make_alias, WhichLaserSlice::np1j00_r, 2);
    m_mg->solve2(np1j00[0], rhs_mg, acoeff_real, acoeff_imag_scalar,
                 m_MG_tolerance_rel, m_MG_tolerance_abs, max_iters, m_MG_verbose);
}

void
MultiLaser::AdvanceSliceFFT (const Fields& fields, const amrex::Real dt, int step)
{

    HIPACE_PROFILE("MultiLaser::AdvanceSliceFFT()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;
    constexpr Complex I(0.,1.);

    const amrex::Real dx = m_laser_geom_3D.CellSize(0);
    const amrex::Real dy = m_laser_geom_3D.CellSize(1);
    const amrex::Real dz = m_laser_geom_3D.CellSize(2);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;
    const amrex::Real k0 = 2.*MathConst::pi/m_lambda0;

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
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
        amrex::Array4<Complex> rhs_fourier_arr = m_rhs_fourier.array();

        Array3<amrex::Real> arr = m_slices.array(mfi);

        constexpr int lev = 0;
        const amrex::FArrayBox& isl_fab = fields.getSlices(lev)[mfi];
        Array3<amrex::Real const> const isl_arr = isl_fab.array();
        const int chi = Comps[WhichSlice::This]["chi"];

        int const Nx = bx.length(0);
        int const Ny = bx.length(1);

        // Get the central point. Useful to get the on-axis phase and calculate kx and ky.
        int const imid = (Nx+1)/2;
        int const jmid = (Ny+1)/2;

        // Calculate phase terms. 0 if !m_use_phase
        amrex::Real tj00 = 0.;
        amrex::Real tjp1 = 0.;
        amrex::Real tjp2 = 0.;

        if (m_use_phase) {
            // Calculate complex arguments (theta) needed
            // Just once, on axis, as done in Wake-T
            // This is done with a reduce operation, returning the sum of the four elements nearest
            // the axis (both real and imag parts, and for the 3 arrays relevant) ...
            amrex::ReduceOps<
                amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<
                amrex::Real, amrex::Real, amrex::Real,
                amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int) -> ReduceTuple
                {
                    using namespace WhichLaserSlice;
                    // Even number of transverse cells: average 2 cells
                    // Odd number of cells: only keep central one
                    const bool do_keep_x = Nx % 2 == 0 ?
                        i == imid-1 || i == imid : i == imid;
                    const bool do_keep_y = Ny % 2 == 0 ?
                        j == jmid-1 || j == jmid : j == jmid;
                    if ( do_keep_x && do_keep_y ) {
                        return {
                            arr(i, j, n00j00_r), arr(i, j, n00j00_i),
                            arr(i, j, n00jp1_r), arr(i, j, n00jp1_i),
                            arr(i, j, n00jp2_r), arr(i, j, n00jp2_i)
                        };
                    } else {
                        return {0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt};
                    }
                });
            // ... and taking the argument of the resulting complex number.
            ReduceTuple hv = reduce_data.value(reduce_op);
            tj00 = std::atan2(amrex::get<1>(hv), amrex::get<0>(hv));
            tjp1 = std::atan2(amrex::get<3>(hv), amrex::get<2>(hv));
            tjp2 = std::atan2(amrex::get<5>(hv), amrex::get<4>(hv));
        }

        amrex::Real dt1 = tj00 - tjp1;
        amrex::Real dt2 = tjp1 - tjp2;
        if (dt1 <-1.5_rt*MathConst::pi) dt1 += 2._rt*MathConst::pi;
        if (dt1 > 1.5_rt*MathConst::pi) dt1 -= 2._rt*MathConst::pi;
        if (dt2 <-1.5_rt*MathConst::pi) dt2 += 2._rt*MathConst::pi;
        if (dt2 > 1.5_rt*MathConst::pi) dt2 -= 2._rt*MathConst::pi;
        Complex exp1 = amrex::exp(I*(tj00-tjp1));
        Complex exp2 = amrex::exp(I*(tj00-tjp2));

        // D_j^n as defined in Benedetti's 2017 paper
        amrex::Real djn = ( -3._rt*dt1 + dt2 ) / (2._rt*dz);
        amrex::ParallelFor(
            bx, 1,
            [=] AMREX_GPU_DEVICE(int i, int j, int, int) noexcept
            {
                using namespace WhichLaserSlice;
                // Transverse Laplacian of real and imaginary parts of A_j^n-1
                amrex::Real lapR, lapI;
                if (step == 0) {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, n00j00_r)+arr(i-1, j, n00j00_r)-2._rt*arr(i, j, n00j00_r))/(dx*dx) +
                        (arr(i, j+1, n00j00_r)+arr(i, j-1, n00j00_r)-2._rt*arr(i, j, n00j00_r))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, n00j00_i)+arr(i-1, j, n00j00_i)-2._rt*arr(i, j, n00j00_i))/(dx*dx) +
                        (arr(i, j+1, n00j00_i)+arr(i, j-1, n00j00_i)-2._rt*arr(i, j, n00j00_i))/(dy*dy) : 0._rt;
                } else {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, nm1j00_r)+arr(i-1, j, nm1j00_r)-2._rt*arr(i, j, nm1j00_r))/(dx*dx) +
                        (arr(i, j+1, nm1j00_r)+arr(i, j-1, nm1j00_r)-2._rt*arr(i, j, nm1j00_r))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, nm1j00_i)+arr(i-1, j, nm1j00_i)-2._rt*arr(i, j, nm1j00_i))/(dx*dx) +
                        (arr(i, j+1, nm1j00_i)+arr(i, j-1, nm1j00_i)-2._rt*arr(i, j, nm1j00_i))/(dy*dy) : 0._rt;
                }
                const Complex lapA = lapR + I*lapI;
                const Complex an00j00 = arr(i, j, n00j00_r) + I * arr(i, j, n00j00_i);
                const Complex anp1jp1 = arr(i, j, np1jp1_r) + I * arr(i, j, np1jp1_i);
                const Complex anp1jp2 = arr(i, j, np1jp2_r) + I * arr(i, j, np1jp2_i);
                Complex rhs;
                if (step == 0) {
                    // First time step: non-centered push to go
                    // from step 0 to step 1 without knowing -1.
                    const Complex an00jp1 = arr(i, j, n00jp1_r) + I * arr(i, j, n00jp1_i);
                    const Complex an00jp2 = arr(i, j, n00jp2_r) + I * arr(i, j, n00jp2_i);
                    rhs =
                        + 8._rt/(c*dt*dz)*(-anp1jp1+an00jp1)*exp1
                        + 2._rt/(c*dt*dz)*(+anp1jp2-an00jp2)*exp2
                        + 2._rt * isl_arr(i,j,chi) * an00j00
                        - lapA
                        + ( -6._rt/(c*dt*dz) + 4._rt*I*djn/(c*dt) + I*4._rt*k0/(c*dt) ) * an00j00;
                } else {
                    const Complex anm1jp1 = arr(i, j, nm1jp1_r) + I * arr(i, j, nm1jp1_i);
                    const Complex anm1jp2 = arr(i, j, nm1jp2_r) + I * arr(i, j, nm1jp2_i);
                    const Complex anm1j00 = arr(i, j, nm1j00_r) + I * arr(i, j, nm1j00_i);
                    rhs =
                        + 4._rt/(c*dt*dz)*(-anp1jp1+anm1jp1)*exp1
                        + 1._rt/(c*dt*dz)*(+anp1jp2-anm1jp2)*exp2
                        - 4._rt/(c*c*dt*dt)*an00j00
                        + 2._rt * isl_arr(i,j,chi) * an00j00
                        - lapA
                        + ( -3._rt/(c*dt*dz) + 2._rt*I*djn/(c*dt) + 2._rt/(c*c*dt*dt) + I*2._rt*k0/(c*dt) ) * anm1j00;
                }
                rhs_arr(i,j,0) = rhs;
            });

        // Transform rhs to Fourier space
#ifdef AMREX_USE_CUDA
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream(m_plan_fwd, stream);
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
        amrex::Real dkx = 2.*MathConst::pi/m_laser_geom_3D.ProbLength(0);
        amrex::Real dky = 2.*MathConst::pi/m_laser_geom_3D.ProbLength(1);
        // acoeff_imag is supposed to be a nx*ny array.
        // For the sake of simplicity, we evaluate it on-axis only.
        const Complex acoeff =
            step == 0 ? 6._rt/(c*dt*dz) - I * 4._rt * ( k0 + djn ) / (c*dt) :
             3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt) - I * 2._rt * ( k0 + djn ) / (c*dt);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // divide rhs_fourier by -(k^2+a)
                amrex::Real kx = (i<imid) ? dkx*i : dkx*(i-Nx);
                amrex::Real ky = (j<jmid) ? dky*j : dky*(j-Ny);
                const Complex inv_k2a = abs(kx*kx + ky*ky + acoeff) > 0. ?
                    1._rt/(kx*kx + ky*ky + acoeff) : 0.;
                rhs_fourier_arr(i,j,k,0) *= -inv_k2a;
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
                using namespace WhichLaserSlice;
                if (i>=imin && i<=imax && j>=jmin && j<=jmax) {
                    arr(i, j, np1j00_r) = sol_arr(i,j,0).real() * inv_numPts;
                    arr(i, j, np1j00_i) = sol_arr(i,j,0).imag() * inv_numPts;
                } else {
                    arr(i, j, np1j00_r) = 0._rt;
                    arr(i, j, np1j00_i) = 0._rt;
                }
            });
    }
}

void
MultiLaser::InitLaserSlice (const amrex::Geometry& geom, const int islice, const int comp)
{
    if (!m_use_laser) return;

    HIPACE_PROFILE("MultiLaser::InitLaserSlice()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;

    // Basic laser parameters and constants
    Complex I(0,1);
    const amrex::Real k0 = 2._rt*MathConst::pi/m_lambda0;

    // Get grid properties
    const auto plo = geom.ProbLoArray();
    amrex::Real const * const dx = geom.CellSize();
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(m_slices, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const & arr = m_slices.array(mfi);

        // Initialize a Gaussian laser envelope on slice islice
        for (int ilaser=0; ilaser<m_nlasers; ilaser++) {
            const auto& laser = m_all_lasers[ilaser];
            const amrex::Real a0 = laser.m_a0;
            const amrex::Real w0 = laser.m_w0;
            const amrex::Real thetayz0 = laser.m_thetayz0;
            const amrex::Real x0 = laser.m_position_mean[0];
            const amrex::Real y0 = laser.m_position_mean[1];
            const amrex::Real z0 = laser.m_position_mean[2];
            const amrex::Real L0 = laser.m_L0;
            const amrex::Real zfoc = laser.m_focal_distance;
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real z = plo[2] + (islice+0.5_rt)*dx_arr[2] - zfoc;
                    
                    const amrex::Real x = (i+0.5_rt)*dx_arr[0]+plo[0]-x0;
                    const amrex::Real y = (j+0.5_rt)*dx_arr[1]+plo[1]-y0;
                    if (thetayz0 != 0._rt) {
                        const amrex::Real xp=std::cos(thetayz0)*x+std::sin(thetayz0)*y;
                        const amrex::Real yp=-std::sin(thetayz0)*x+std::cos(thetayz0)*y;
                        x=xp;
                        y=yp;
                    }
                    // For first laser, setval to 0.
                    if (ilaser == 0) {
                        arr(i, j, k, comp ) = 0._rt;
                        arr(i, j, k, comp + 1 ) = 0._rt;
                    }

                    // Compute envelope for time step 0
                    Complex diffract_factor = 1._rt + I * z * 2._rt/( k0 * w0 * w0 );
                    Complex inv_complex_waist_2 = 1._rt /( w0 * w0 * diffract_factor );
                    Complex prefactor = a0/diffract_factor;
                    Complex time_exponent = (z-z0+zfoc)*(z-z0+zfoc)/(L0*L0);
                    Complex stcfactor = prefactor * amrex::exp( - time_exponent );
                    Complex exp_argument = - ( x*x + y*y ) * inv_complex_waist_2;
                    Complex envelope = stcfactor * amrex::exp( exp_argument );
                    arr(i, j, k, comp ) += envelope.real();
                    arr(i, j, k, comp + 1 ) += envelope.imag();
                }
                );
        }
    }
}
