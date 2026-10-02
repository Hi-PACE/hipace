/* Copyright 2022-2025
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers
 * License: BSD-3-Clause-LBNL
 */

#include "Helmholtz.H"
#include "utils/Constants.H"
#include "fields/Fields.H"
#include "Hipace.H"
#include "particles/plasma/MultiPlasma.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/InsituUtil.H"
#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "particles/particles_utils/ShapeFactors.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif

#include <AMReX_GpuComplex.H>

void
Helmholtz::ReadParameters ()
{
    amrex::ParmParse pp("helmholtz");
    queryWithParser(pp, "use_helmholtz", m_use_helmholtz);

    if (!m_use_helmholtz) return;

    getWithParser(pp, "mode", m_mode);
    AMREX_ALWAYS_ASSERT(m_mode == "envelope" || m_mode == "full_field");
    if (ModeIsEnvelope()) {
        getWithParser(pp, "lambda0", m_lambda0);
        queryWithParser(pp, "use_mg", m_use_mg);
        m_k0 = 2.*MathConst::pi/m_lambda0;
        queryWithParser(pp, "use_phase", m_use_phase);
        queryWithParser(pp, "zfilter_source", m_zfilter_source);
    }
    queryWithParser(pp, "interp_order", m_interp_order);
    AMREX_ALWAYS_ASSERT(m_interp_order <= 3 && m_interp_order >= 0);
    queryWithParser(pp, "insitu_period", m_insitu_period.m_func_str);
    m_insitu_period.compile();
    queryWithParser(pp, "centered_dz", m_centered_dz);
    queryWithParser(pp, "add_dx_jz", m_add_dx_jz);
    queryWithParser(pp, "add_dz_jx", m_add_dz_jx);
    queryWithParser(pp, "interp_z", m_interp_z);
    queryWithParser(pp, "first_order", m_first_order);

    std::string profile_real_str = "0.";
    std::string profile_imag_str = "0.";
    queryWithParser(pp, "field_real(x,y,z)", profile_real_str);
    queryWithParser(pp, "field_imag(x,y,z)", profile_imag_str);
    m_profile_real = makeFunctionWithParser<3>( profile_real_str, m_parser_lr, {"x", "y", "z"});
    m_profile_imag = makeFunctionWithParser<3>( profile_imag_str, m_parser_li, {"x", "y", "z"});

    m_insitu_file_prefix = Hipace::m_output_folder + "/insitu";
    const bool set_file_prefix =  queryWithParser(pp, "insitu_file_prefix", m_insitu_file_prefix);
    if (set_file_prefix) {
        amrex::Print() <<
            "It is recommended to use hipace.output_folder instead of lasers.insitu_file_prefix\n";
    }
}

void
Helmholtz::MakeHelmholtzGeometry (const amrex::Geometry& field_geom_3D)
{
    if (!m_use_helmholtz) return;
    amrex::ParmParse pp("helmholtz");

    // use field_geom_3D as the default
    std::array<int, 2> n_cells_helmholtz {field_geom_3D.Domain().length(0),
                                          field_geom_3D.Domain().length(1)};
    std::array<amrex::Real, 3> patch_lo_helmholtz {
        field_geom_3D.ProbDomain().lo(0),
        field_geom_3D.ProbDomain().lo(1),
        field_geom_3D.ProbDomain().lo(2)};
    std::array<amrex::Real, 3> patch_hi_helmholtz {
        field_geom_3D.ProbDomain().hi(0),
        field_geom_3D.ProbDomain().hi(1),
        field_geom_3D.ProbDomain().hi(2)};

    // get parameters from user input
    queryWithParser(pp, "n_cell", n_cells_helmholtz);
    queryWithParser(pp, "patch_lo", patch_lo_helmholtz);
    queryWithParser(pp, "patch_hi", patch_hi_helmholtz);

    // round zeta lo and hi to full cells
    const amrex::Real pos_offset_z = GetPosOffset(2, field_geom_3D, field_geom_3D.Domain());

    const int zeta_lo = std::max( field_geom_3D.Domain().smallEnd(2),
        int(amrex::Math::round((patch_lo_helmholtz[2] - pos_offset_z) * field_geom_3D.InvCellSize(2)))
    );

    const int zeta_hi = std::min( field_geom_3D.Domain().bigEnd(2),
        int(amrex::Math::round((patch_hi_helmholtz[2] - pos_offset_z) * field_geom_3D.InvCellSize(2)))
    );

    patch_lo_helmholtz[2] = (zeta_lo-0.5)*field_geom_3D.CellSize(2) + pos_offset_z;
    patch_hi_helmholtz[2] = (zeta_hi+0.5)*field_geom_3D.CellSize(2) + pos_offset_z;

    // make the boxes
    const amrex::Box domain_3D_helmholtz{amrex::IntVect(0, 0, zeta_lo),
        amrex::IntVect(n_cells_helmholtz[0]-1, n_cells_helmholtz[1]-1, zeta_hi)};

    const amrex::RealBox real_box(patch_lo_helmholtz, patch_hi_helmholtz);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(real_box.volume() > 0., "Helmholtz box must have positive volume");

    // make the geometry, slice box and ba and dm
    m_helmholtz_geom_3D.define(domain_3D_helmholtz, real_box, amrex::CoordSys::cartesian, {0, 0, 0});

    m_slice_box = domain_3D_helmholtz;
    m_slice_box.setSmall(2, 0);
    m_slice_box.setBig(2, 0);

    m_helmholtz_slice_ba.define(m_slice_box);
    m_helmholtz_slice_dm.define(amrex::Vector<int>({amrex::ParallelDescriptor::MyProc()}));
}

void
Helmholtz::InitData ()
{
    if (!m_use_helmholtz) return;

    HIPACE_PROFILE("Helmholtz::InitData()");

    HelmholtzComps.multi_emplace(N_HelmholtzComps,
        "Ex_nm1j00",  "Ex_nm1jp1",  "Ex_nm1jp2",  "Ex_nm1jm1",
        "Ex_n00j00",  "Ex_n00jp1",  "Ex_n00jp2",  "Ex_n00jm1",
        "Ex_np1j00",  "Ei_np1j00",
        "Ex_np1jp1",  "Ei_np1jp1",
        "Ex_np1jp2",  "Ei_np1jp2",
        "jx_n00jm1",  "jx_n00j00",  "jx_n00jp1",  "jx_n00jp2",
        "jz_n00jm1",  "jz_n00j00",  "jz_n00jp1",
        "rho_n00jm1", "rho_n00j00", "rho_n00jp1",
        "Ei_nm1j00",  "Ei_nm1jp1",  "Ei_nm1jp2",  "Ei_nm1jm1",
        "Ei_n00j00",  "Ei_n00jp1",  "Ei_n00jp2",  "Ei_n00jm1"
    );

    // Alloc 2D slices
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = (Hipace::m_depos_order_xy + 1) / 2 + 1;
    m_slices_nguards = {nguards_xy, nguards_xy, 0};
    m_slices.define(
        m_helmholtz_slice_ba, m_helmholtz_slice_dm, WhichHelmholtzSlice::N, m_slices_nguards,
        amrex::MFInfo().SetArena(amrex::The_Arena()));
    m_slices.setVal(0.0);

    m_sol.resize(m_slice_box, 1, amrex::The_Arena());
    m_rhs.resize(m_slice_box, 1, amrex::The_Arena());
    m_rhs_fourier.resize(m_slice_box, 1, amrex::The_Arena());

    if (m_use_mg) {
        // Initialize Multigrid solver
        // need one ghost cell for 2^n-1 MG solve
        m_mg_acoeff_real.resize(amrex::grow(m_slice_box, amrex::IntVect{1, 1, 0}), 1, amrex::The_Arena());
        m_rhs_mg.resize(amrex::grow(m_slice_box, amrex::IntVect{1, 1, 0}), 2, amrex::The_Arena());
    } else {
        // Create FFT plans
        amrex::IntVect fft_size = m_slice_box.length();

        std::size_t fwd_area = m_forward_fft.Initialize(FFTType::C2C_2D_fwd, fft_size[0], fft_size[1]);
        std::size_t bkw_area = m_backward_fft.Initialize(FFTType::C2C_2D_bkw, fft_size[0], fft_size[1]);

        // Allocate work area for both FFTs
        m_fft_work_area.resize(std::max(fwd_area, bkw_area));

        m_forward_fft.SetBuffers(m_rhs.dataPtr(), m_rhs_fourier.dataPtr(), m_fft_work_area.dataPtr());
        m_backward_fft.SetBuffers(m_rhs_fourier.dataPtr(), m_sol.dataPtr(), m_fft_work_area.dataPtr());
    }

    if (m_insitu_period.isNonZero()) {
#ifdef HIPACE_USE_OPENPMD
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_insitu_file_prefix !=
            Hipace::GetInstance().m_openpmd_writer.m_file_prefix,
            "Must choose a different field insitu file prefix compared to the full diagnostics");
#endif
        // Allocate memory for in-situ diagnostics
        m_insitu_rdata.resize(m_helmholtz_geom_3D.Domain().length(2)*m_insitu_nrp, 0.);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
        m_insitu_cdata.resize(m_helmholtz_geom_3D.Domain().length(2)*m_insitu_ncp, 0.);
    }
}

void
Helmholtz::InitSliceEnvelope (const int islice, const int comp)
{
    if (!UseHelmholtz(islice)) return;

    HIPACE_PROFILE("Helmholtz::InitSliceEnvelope()");

    InitHelmholtzSlice(islice, comp);
}

void
Helmholtz::ShiftHelmholtzSlices (const int islice)
{
    if (!UseHelmholtz(islice)) return;

    HIPACE_PROFILE("Helmholtz::ShiftHelmholtzSlices()");

    using namespace amrex::literals;
    bool mode_is_envelope = ModeIsEnvelope();

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        Array3<amrex::Real> arr = m_slices.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
        {
            using namespace WhichHelmholtzSlice;
            // Shift slices of step n-1
            arr(i, j, Ex_nm1jp2) = arr(i, j, Ex_nm1jp1);
            arr(i, j, Ex_nm1jp1) = arr(i, j, Ex_nm1j00);
            arr(i, j, Ex_nm1j00) = arr(i, j, Ex_nm1jm1);
            arr(i, j, Ex_nm1jm1) = 0._rt;
            // Shift slices of step n
            arr(i, j, Ex_n00jp2) = arr(i, j, Ex_n00jp1);
            arr(i, j, Ex_n00jp1) = arr(i, j, Ex_n00j00);
            arr(i, j, Ex_n00j00) = arr(i, j, Ex_n00jm1);
            arr(i, j, Ex_n00jm1) = 0._rt;
            // Shift slices of step n+1
            arr(i, j, Ex_np1jp2) = arr(i, j, Ex_np1jp1);
            arr(i, j, Ex_np1jp1) = arr(i, j, Ex_np1j00);
            // np1j00_r will be computed by AdvanceSlice

            arr(i, j, jx_n00jp2) = arr(i, j, jx_n00jp1);
            arr(i, j, jx_n00jp1) = arr(i, j, jx_n00j00);
            arr(i, j, jx_n00j00) = arr(i, j, jx_n00jm1);
            arr(i, j, jx_n00jm1) = 0._rt;

            arr(i, j, jz_n00jp1) = arr(i, j, jz_n00j00);
            arr(i, j, jz_n00j00) = arr(i, j, jz_n00jm1);
            arr(i, j, jz_n00jm1) = 0._rt;

            arr(i, j, rho_n00jp1) = arr(i, j, rho_n00j00);
            arr(i, j, rho_n00j00) = arr(i, j, rho_n00jm1);
            arr(i, j, rho_n00jm1) = 0._rt;

            if (mode_is_envelope) {
                // Shift slices of step n-1
                arr(i, j, Ei_nm1jp2) = arr(i, j, Ei_nm1jp1);
                arr(i, j, Ei_nm1jp1) = arr(i, j, Ei_nm1j00);
                arr(i, j, Ei_nm1j00) = arr(i, j, Ei_nm1jm1);
                arr(i, j, Ei_nm1jm1) = 0._rt;
                // Shift slices of step n
                arr(i, j, Ei_n00jp2) = arr(i, j, Ei_n00jp1);
                arr(i, j, Ei_n00jp1) = arr(i, j, Ei_n00j00);
                arr(i, j, Ei_n00j00) = arr(i, j, Ei_n00jm1);
                arr(i, j, Ei_n00jm1) = 0._rt;
                // Shift slices of step n+1
                arr(i, j, Ei_np1jp2) = arr(i, j, Ei_np1jp1);
                arr(i, j, Ei_np1jp1) = arr(i, j, Ei_np1j00);
            }
        });
    }
}

void
Helmholtz::AdvanceSlice (const int islice, amrex::Real dt, int step)
{

    if (!UseHelmholtz(islice)) return;

    if (ModeIsEnvelope()) {
        if (m_use_mg) {
            AdvanceSliceMGEnvelope(dt, step);
        } else {
            AdvanceSliceFFTEnvelope(dt, step);
        }
    } else {
        AdvanceSliceFFT(dt, step);
    }
}

void
Helmholtz::AdvanceSliceFFT (const amrex::Real dt, int step)
{

    HIPACE_PROFILE("Helmholtz::AdvanceSliceFFT()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;

    const amrex::Real dx = m_helmholtz_geom_3D.CellSize(0);
    const amrex::Real dy = m_helmholtz_geom_3D.CellSize(1);
    const amrex::Real dz = m_helmholtz_geom_3D.CellSize(2);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;
    const bool centered_dz = m_centered_dz;
    const bool add_dx_jz = m_add_dx_jz;
    const bool add_dz_jx = m_add_dz_jx;
    const bool first_order = FirstOrder(step);

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

        int const Nx = bx.length(0);
        int const Ny = bx.length(1);

        // Get the central point. Useful to get the on-axis phase and calculate kx and ky.
        int const imid = (Nx+1)/2;
        int const jmid = (Ny+1)/2;

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                using namespace WhichHelmholtzSlice;
                // Transverse Laplacian of A_j^n-1
                amrex::Real lap;
                if (first_order) {
                    lap = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j,Ex_n00j00)+arr(i-1, j,Ex_n00j00) - 2._rt*arr(i,j,Ex_n00j00))/(dx*dx) +
                        (arr(i, j+1,Ex_n00j00)+arr(i, j-1,Ex_n00j00) - 2._rt*arr(i,j,Ex_n00j00))/(dy*dy) : 0._rt;
                } else {
                    lap = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j,Ex_nm1j00)+arr(i-1, j,Ex_nm1j00) - 2._rt*arr(i,j,Ex_nm1j00))/(dx*dx) +
                        (arr(i, j+1,Ex_nm1j00)+arr(i, j-1,Ex_nm1j00) - 2._rt*arr(i,j,Ex_nm1j00))/(dy*dy) : 0._rt;
                }
                const amrex::Real an00j00 = arr(i, j, Ex_n00j00);
                const amrex::Real anp1jp1 = arr(i, j, Ex_np1jp1);
                const amrex::Real anp1jp2 = arr(i, j, Ex_np1jp2);
                amrex::Real rhs;
                if (first_order) {
                    // First time step: non-centered push to go
                    // from step 0 to step 1 without knowing -1.
                    const amrex::Real an00jp1 = arr(i, j, Ex_n00jp1);
                    const amrex::Real an00jp2 = arr(i, j, Ex_n00jp2);
                    rhs =
                        + 8._rt/(c*dt*dz)*(-anp1jp1+an00jp1)
                        + 2._rt/(c*dt*dz)*(+anp1jp2-an00jp2)
                        - lap
                        + ( -6._rt/(c*dt*dz) ) * an00j00;
                } else {
                    const amrex::Real anm1jp1 = arr(i, j, Ex_nm1jp1);
                    const amrex::Real anm1jp2 = arr(i, j, Ex_nm1jp2);
                    const amrex::Real anm1j00 = arr(i, j, Ex_nm1j00);
                    rhs =
                        + 4._rt/(c*dt*dz)*(-anp1jp1+anm1jp1)
                        + 1._rt/(c*dt*dz)*(+anp1jp2-anm1jp2)
                        - 4._rt/(c*c*dt*dt)*an00j00
                        - lap
                        + ( -3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt) ) * anm1j00;
                }

                if (add_dz_jx) {
                    const amrex::Real dz_jx = centered_dz ?
                        0.5_rt * (arr(i, j, jx_n00jp1) - arr(i, j, jx_n00jm1)) / dz :
                        0.5_rt * (- 3._rt*arr(i, j, jx_n00j00)
                                  + 4._rt*arr(i, j, jx_n00jp1)
                                  - arr(i, j, jx_n00jp2) ) / dz;
                    rhs -= 2._rt * phc.mu0 * c * dz_jx;
                }

                if (add_dx_jz) {
                    const amrex::Real dx_jz = i>imin && i<imax ?
                        (arr(i+1, j, jz_n00j00) - arr(i-1, j, jz_n00j00)) / (2._rt*dx) : 0._rt;
                    rhs += 2._rt * phc.mu0 * c * dx_jz;
                }

                rhs_arr(i,j,0) = rhs;
            });

        // Transform rhs to Fourier space
        m_forward_fft.Execute();


        // Multiply by appropriate factors in Fourier space
        amrex::Real dkx = 2.*MathConst::pi/m_helmholtz_geom_3D.ProbLength(0);
        amrex::Real dky = 2.*MathConst::pi/m_helmholtz_geom_3D.ProbLength(1);
        const amrex::Real acoeff = first_order ? 6._rt/(c*dt*dz) : 3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // divide rhs_fourier by -(k^2+a)
                amrex::Real kx = (i<imid) ? dkx*i : dkx*(i-Nx);
                amrex::Real ky = (j<jmid) ? dky*j : dky*(j-Ny);
                const amrex::Real inv_k2a = std::abs(kx*kx + ky*ky + acoeff) > 0. ? 1._rt/(kx*kx + ky*ky + acoeff) : 0.;
                rhs_fourier_arr(i,j,k,0) *= -inv_k2a;
            });

        // Transform rhs to Fourier space to get solution in sol
        m_backward_fft.Execute();

        // Normalize and store solution in np1j00[0]. Guard cells are filled with 0s.
        amrex::Box grown_bx = bx;
        grown_bx.grow(m_slices_nguards);
        const amrex::Real inv_numPts = 1./bx.numPts();
        amrex::ParallelFor(
            grown_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                using namespace WhichHelmholtzSlice;
                if (i>=imin && i<=imax && j>=jmin && j<=jmax) {
                    arr(i, j, Ex_np1j00) = sol_arr(i,j,0).real() * inv_numPts;
                } else {
                    arr(i, j, Ex_np1j00) = 0._rt;
                }
            });
    }
}

void
Helmholtz::AdvanceSliceMGEnvelope (amrex::Real dt, int step)
{

    HIPACE_PROFILE("Helmholtz::AdvanceSliceMGEnvelope()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;
    constexpr Complex I(0.,1.);

    const amrex::Real dx = m_helmholtz_geom_3D.CellSize(0);
    const amrex::Real dy = m_helmholtz_geom_3D.CellSize(1);
    const amrex::Real dz = m_helmholtz_geom_3D.CellSize(2);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;
    const amrex::Real k0 = m_k0;
    const bool do_avg_rhs = m_MG_average_rhs;
    const bool zfilter_source = m_zfilter_source;
    const bool first_order = FirstOrder(step);

    amrex::Real acoeff_real_scalar = 0._rt;
    amrex::Real acoeff_imag_scalar = 0._rt;

    amrex::Real djn {0.};

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        const int imin = bx.smallEnd(0);
        const int imax = bx.bigEnd  (0);
        const int jmin = bx.smallEnd(1);
        const int jmax = bx.bigEnd  (1);

        Array3<amrex::Real> arr = m_slices.array(mfi);
        Array3<amrex::Real> rhs_mg_arr = m_rhs_mg.array();
        Array3<amrex::Real> acoeff_real_arr = m_mg_acoeff_real.array();

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
                    using namespace WhichHelmholtzSlice;
                    // Even number of transverse cells: average 2 cells
                    // Odd number of cells: only keep central one
                    const bool do_keep_x = Nx % 2 == 0 ?
                        i == imid-1 || i == imid : i == imid;
                    const bool do_keep_y = Ny % 2 == 0 ?
                        j == jmid-1 || j == jmid : j == jmid;
                    if ( do_keep_x && do_keep_y ) {
                        return {
                            arr(i, j, Ex_n00j00), arr(i, j, Ei_n00j00),
                            arr(i, j, Ex_n00jp1), arr(i, j, Ei_n00jp1),
                            arr(i, j, Ex_n00jp2), arr(i, j, Ei_n00jp2)
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
        acoeff_real_scalar = first_order ? 6._rt/(c*dt*dz)
            : 3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt);
        acoeff_imag_scalar = first_order ? -4._rt * ( k0 + djn ) / (c*dt)
            : -2._rt * ( k0 + djn ) / (c*dt);

        amrex::ParallelFor(
            to2D(bx),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                using namespace WhichHelmholtzSlice;
                // Transverse Laplacian of real and imaginary parts of A_j^n-1
                amrex::Real lapR, lapI;
                if (first_order) {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ex_n00j00)+arr(i-1, j, Ex_n00j00)-2._rt*arr(i, j, Ex_n00j00))/(dx*dx) +
                        (arr(i, j+1, Ex_n00j00)+arr(i, j-1, Ex_n00j00)-2._rt*arr(i, j, Ex_n00j00))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ei_n00j00)+arr(i-1, j, Ei_n00j00)-2._rt*arr(i, j, Ei_n00j00))/(dx*dx) +
                        (arr(i, j+1, Ei_n00j00)+arr(i, j-1, Ei_n00j00)-2._rt*arr(i, j, Ei_n00j00))/(dy*dy) : 0._rt;
                } else {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ex_nm1j00)+arr(i-1, j, Ex_nm1j00)-2._rt*arr(i, j, Ex_nm1j00))/(dx*dx) +
                        (arr(i, j+1, Ex_nm1j00)+arr(i, j-1, Ex_nm1j00)-2._rt*arr(i, j, Ex_nm1j00))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ei_nm1j00)+arr(i-1, j, Ei_nm1j00)-2._rt*arr(i, j, Ei_nm1j00))/(dx*dx) +
                        (arr(i, j+1, Ei_nm1j00)+arr(i, j-1, Ei_nm1j00)-2._rt*arr(i, j, Ei_nm1j00))/(dy*dy) : 0._rt;
                }
                const Complex lapA = lapR + I*lapI;
                const Complex an00j00 = arr(i, j, Ex_n00j00) + I * arr(i, j, Ei_n00j00);
                const Complex anp1jp1 = arr(i, j, Ex_np1jp1) + I * arr(i, j, Ei_np1jp1);
                const Complex anp1jp2 = arr(i, j, Ex_np1jp2) + I * arr(i, j, Ei_np1jp2);
                const amrex::Real chi = arr(i, j, jx_n00j00);
                const Complex source = zfilter_source ?
                    0.50_rt * ( arr(i, j, jz_n00j00) + I * arr(i, j, rho_n00j00) ) +
                    0.25_rt * ( arr(i, j, jz_n00jp1) + I * arr(i, j, rho_n00jp1) ) +
                    0.25_rt * ( arr(i, j, jz_n00jm1) + I * arr(i, j, rho_n00jm1) )
                    :           arr(i, j, jz_n00j00) + I * arr(i, j, rho_n00j00);
                // 1/ga      cos(t)/ga sin(t)/ga
                // jx_n00j00 jz_n00j00 rho_n00j00
                acoeff_real_arr(i,j,0) = do_avg_rhs ?
                    acoeff_real_scalar + chi : acoeff_real_scalar;

                Complex rhs;
                if (first_order) {
                    // First time step: non-centered push to go
                    // from step 0 to step 1 without knowing -1.
                    const Complex an00jp1 = arr(i, j, Ex_n00jp1) + I * arr(i, j, Ei_n00jp1);
                    const Complex an00jp2 = arr(i, j, Ex_n00jp2) + I * arr(i, j, Ei_n00jp2);
                    rhs =
                        + 8._rt/(c*dt*dz)*(-anp1jp1+an00jp1)*exp1
                        + 2._rt/(c*dt*dz)*(+anp1jp2-an00jp2)*exp2
                        - lapA
                        + ( -6._rt/(c*dt*dz) + 4._rt*I*djn/(c*dt) + I*4._rt*k0/(c*dt) ) * an00j00;
                    if (do_avg_rhs) {
                        rhs += chi * an00j00;
                    } else {
                        rhs += chi * an00j00 * 2._rt;
                    }
                } else {
                    const Complex anm1jp1 = arr(i, j, Ex_nm1jp1) + I * arr(i, j, Ei_nm1jp1);
                    const Complex anm1jp2 = arr(i, j, Ex_nm1jp2) + I * arr(i, j, Ei_nm1jp2);
                    const Complex anm1j00 = arr(i, j, Ex_nm1j00) + I * arr(i, j, Ei_nm1j00);
                    rhs =
                        + 4._rt/(c*dt*dz)*(-anp1jp1+anm1jp1)*exp1
                        + 1._rt/(c*dt*dz)*(+anp1jp2-anm1jp2)*exp2
                        - 4._rt/(c*c*dt*dt)*an00j00
                        - lapA
                        + ( -3._rt/(c*dt*dz) + 2._rt*I*djn/(c*dt) + 2._rt/(c*c*dt*dt) + I*2._rt*k0/(c*dt) ) * anm1j00;
                    if (do_avg_rhs) {
                        rhs += chi * anm1j00;
                    } else {
                        rhs += chi * an00j00 * 2._rt;
                    }
                }
                rhs += 2._rt * source; // usual factor of 2 from discr. of dzeta in lhs
                rhs_mg_arr(i,j,0) = rhs.real();
                rhs_mg_arr(i,j,1) = rhs.imag();
            });
    }

    if (!m_mg) {
        m_mg = std::make_unique<hpmg::MultiGrid>(m_helmholtz_geom_3D.CellSize(0),
                                                 m_helmholtz_geom_3D.CellSize(1),
                                                 m_slices.boxArray()[0], 2);
    }

    const int max_iters = 200;
    amrex::MultiFab np1j00 (m_slices, amrex::make_alias, WhichHelmholtzSlice::Ex_np1j00, 2);
    m_mg->solve2(np1j00[0], m_rhs_mg, m_mg_acoeff_real, acoeff_imag_scalar,
                 m_MG_tolerance_rel, m_MG_tolerance_abs, max_iters, m_MG_verbose);
}

void
Helmholtz::AdvanceSliceFFTEnvelope (const amrex::Real dt, int step)
{

    HIPACE_PROFILE("Helmholtz::AdvanceSliceFFTEnvelope()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;
    constexpr Complex I(0.,1.);

    const amrex::Real dx = m_helmholtz_geom_3D.CellSize(0);
    const amrex::Real dy = m_helmholtz_geom_3D.CellSize(1);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;
    const amrex::Real k0 = m_k0;
    const bool zfilter_source = m_zfilter_source;
    const bool first_order = FirstOrder(step);

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
        Array2<Complex> rhs_fourier_arr = m_rhs_fourier.array();

        Array3<amrex::Real> arr = m_slices.array(mfi);

        int const Nx = bx.length(0);
        int const Ny = bx.length(1);

        // Get the central point. Useful to get the on-axis phase and calculate kx and ky.
        int const imid = (Nx+1)/2;
        int const jmid = (Ny+1)/2;

        amrex::ParallelFor(
            to2D(bx),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                using namespace WhichHelmholtzSlice;
                // Transverse Laplacian of real and imaginary parts of A_j^n-1
                amrex::Real lapR, lapI;
                if (first_order) {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ex_n00j00)+arr(i-1, j, Ex_n00j00)-2._rt*arr(i, j, Ex_n00j00))/(dx*dx) +
                        (arr(i, j+1, Ex_n00j00)+arr(i, j-1, Ex_n00j00)-2._rt*arr(i, j, Ex_n00j00))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ei_n00j00)+arr(i-1, j, Ei_n00j00)-2._rt*arr(i, j, Ei_n00j00))/(dx*dx) +
                        (arr(i, j+1, Ei_n00j00)+arr(i, j-1, Ei_n00j00)-2._rt*arr(i, j, Ei_n00j00))/(dy*dy) : 0._rt;
                } else {
                    lapR = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ex_nm1j00)+arr(i-1, j, Ex_nm1j00)-2._rt*arr(i, j, Ex_nm1j00))/(dx*dx) +
                        (arr(i, j+1, Ex_nm1j00)+arr(i, j-1, Ex_nm1j00)-2._rt*arr(i, j, Ex_nm1j00))/(dy*dy) : 0._rt;
                    lapI = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, Ei_nm1j00)+arr(i-1, j, Ei_nm1j00)-2._rt*arr(i, j, Ei_nm1j00))/(dx*dx) +
                        (arr(i, j+1, Ei_nm1j00)+arr(i, j-1, Ei_nm1j00)-2._rt*arr(i, j, Ei_nm1j00))/(dy*dy) : 0._rt;
                }
                const Complex lapA = lapR + I*lapI;
                const Complex an00j00 = arr(i, j, Ex_n00j00) + I * arr(i, j, Ei_n00j00);
                const amrex::Real chi = arr(i, j, jx_n00j00);
                const Complex source = zfilter_source ?
                    0.50_rt * ( arr(i, j, jz_n00j00) + I * arr(i, j, rho_n00j00) ) +
                    0.25_rt * ( arr(i, j, jz_n00jp1) + I * arr(i, j, rho_n00jp1) ) +
                    0.25_rt * ( arr(i, j, jz_n00jm1) + I * arr(i, j, rho_n00jm1) )
                    :           arr(i, j, jz_n00j00) + I * arr(i, j, rho_n00j00);
                Complex rhs;
                if (first_order) {
                    // First time step: non-centered push to go
                    // from step 0 to step 1 without knowing -1.
                    rhs =
                        + 2._rt * chi * an00j00
                        - lapA
                        + I*4._rt*k0/(c*dt) * an00j00;
                } else {
                    const Complex anm1j00 = arr(i, j, Ex_nm1j00) + I * arr(i, j, Ei_nm1j00);
                    rhs =
                        + 2._rt * chi * an00j00
                        - lapA
                        + I*2._rt*k0/(c*dt) * anm1j00;
                }
                rhs += 2._rt * source; // usual factor of 2 from discr. of dzeta in lhs
                rhs_arr(i,j,0) = rhs;
            });

        // Transform rhs to Fourier space
        m_forward_fft.Execute();

        // Multiply by appropriate factors in Fourier space
        amrex::Real dkx = 2.*MathConst::pi/m_helmholtz_geom_3D.ProbLength(0);
        amrex::Real dky = 2.*MathConst::pi/m_helmholtz_geom_3D.ProbLength(1);
        // acoeff_imag is supposed to be a nx*ny array.
        // For the sake of simplicity, we evaluate it on-axis only.
        const Complex acoeff =
            first_order ? - I * 4._rt * k0 / (c*dt) : - I * 2._rt * k0 / (c*dt);
        amrex::ParallelFor(
            to2D(bx),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept {
                // divide rhs_fourier by -(k^2+a)
                amrex::Real kx = (i<imid) ? dkx*i : dkx*(i-Nx);
                amrex::Real ky = (j<jmid) ? dky*j : dky*(j-Ny);
                const Complex inv_k2a = abs(kx*kx + ky*ky + acoeff) > 0. ?
                    1._rt/(kx*kx + ky*ky + acoeff) : 0.;
                rhs_fourier_arr(i,j) *= -inv_k2a;
            });

        // Transform rhs to Fourier space to get solution in sol
        m_backward_fft.Execute();

        // Normalize and store solution in np1j00[0]. Guard cells are filled with 0s.
        amrex::Box grown_bx = bx;
        grown_bx.grow(m_slices_nguards);
        const amrex::Real inv_numPts = 1./bx.numPts();
        amrex::ParallelFor(
            to2D(grown_bx),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept {
                using namespace WhichHelmholtzSlice;
                if (i>=imin && i<=imax && j>=jmin && j<=jmax) {
                    arr(i, j, Ex_np1j00) = sol_arr(i,j,0).real() * inv_numPts;
                    arr(i, j, Ei_np1j00) = sol_arr(i,j,0).imag() * inv_numPts;
                } else {
                    arr(i, j, Ex_np1j00) = 0._rt;
                    arr(i, j, Ei_np1j00) = 0._rt;
                }
            });
    }
}

void
Helmholtz::InitHelmholtzSlice (const int islice, const int comp)
{
    HIPACE_PROFILE("Helmholtz::InitHelmholtzSlice()");

    using namespace amrex::literals;

    const amrex::Real poff_x = GetPosOffset(0, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::Real poff_y = GetPosOffset(1, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::Real poff_z = GetPosOffset(2, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::GpuArray<amrex::Real, 3> dx_arr = m_helmholtz_geom_3D.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(m_slices, DfltMfiTlng); mfi.isValid(); ++mfi ){
        amrex::Array4<amrex::Real> const & arr = m_slices.array(mfi);
        auto profile_real = m_profile_real;
        auto profile_imag = m_profile_imag;
        // Initialize a Gaussian helmholtz envelope on slice islice
        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // arr(i, j, k, comp ) = 0._rt;
                const amrex::Real x = i * dx_arr[0] + poff_x;
                const amrex::Real y = j * dx_arr[1] + poff_y;
                const amrex::Real z = islice * dx_arr[2] + poff_z;
                if (comp == WhichHelmholtzSlice::Ex_n00jm1 ||
                    comp == WhichHelmholtzSlice::Ex_n00j00) {
                    arr(i, j, k, comp ) = profile_real(x,y,z);
                }
                if (comp == WhichHelmholtzSlice::Ei_n00jm1 ||
                    comp == WhichHelmholtzSlice::Ei_n00j00) {
                    arr(i, j, k, comp ) = profile_imag(x,y,z);
                }
            }
            );
    }
}

void
Helmholtz::InSituComputeDiags (int step, amrex::Real time, int islice, bool is_last_step)
{
    if (!UseHelmholtz(islice)) return;
    if (!m_insitu_period.doDiagnostics(step, time, is_last_step)) return;
    HIPACE_PROFILE("Helmholtz::InSituComputeDiags()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;

    AMREX_ALWAYS_ASSERT(m_insitu_rdata.size()>0 && m_insitu_sum_rdata.size()>0 &&
                        m_insitu_cdata.size()>0);

    const int nslices = m_helmholtz_geom_3D.Domain().length(2);
    const int helmholtz_slice = islice - m_helmholtz_geom_3D.Domain().smallEnd(2);
    const amrex::Real poff_x = GetPosOffset(0, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::Real poff_y = GetPosOffset(1, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::Real dx = m_helmholtz_geom_3D.CellSize(0);
    const amrex::Real dy = m_helmholtz_geom_3D.CellSize(1);
    const amrex::Real dxdydz = dx * dy * m_helmholtz_geom_3D.CellSize(2);
    const bool mode_is_envelope = ModeIsEnvelope();

    const int xmid_lo = m_helmholtz_geom_3D.Domain().smallEnd(0) + (m_helmholtz_geom_3D.Domain().length(0) - 1) / 2;
    const int xmid_hi = m_helmholtz_geom_3D.Domain().smallEnd(0) + (m_helmholtz_geom_3D.Domain().length(0)) / 2;
    const int ymid_lo = m_helmholtz_geom_3D.Domain().smallEnd(1) + (m_helmholtz_geom_3D.Domain().length(1) - 1) / 2;
    const int ymid_hi = m_helmholtz_geom_3D.Domain().smallEnd(1) + (m_helmholtz_geom_3D.Domain().length(1)) / 2;
    const amrex::Real mid_factor = (xmid_lo == xmid_hi ? 1._rt : 0.5_rt)
                                 * (ymid_lo == ymid_hi ? 1._rt : 0.5_rt);

    amrex::TypeMultiplier<amrex::ReduceOps, amrex::ReduceOpMax, amrex::ReduceOpSum[m_insitu_nrp-1+m_insitu_ncp]> reduce_op;
    amrex::TypeMultiplier<amrex::ReduceData, amrex::Real[m_insitu_nrp], Complex[m_insitu_ncp]> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ) {
        Array3<amrex::Real const> const arr = m_slices.const_array(mfi);
        reduce_op.eval(
            mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int) -> ReduceTuple
            {
                using namespace WhichHelmholtzSlice;
                const amrex::Real ex = arr(i,j, Ex_n00j00);
                const amrex::Real ei = mode_is_envelope ? arr(i,j, Ei_n00j00) : 0._rt;
                const amrex::Real aabssq = ex*ex + ei*ei;
                const amrex::Real s1 = arr(i,j, jz_n00j00);
                const amrex::Real s2 = arr(i,j, rho_n00j00);

                const amrex::Real x = i * dx + poff_x;
                const amrex::Real y = j * dy + poff_y;

                const bool is_on_axis = (i==xmid_lo || i==xmid_hi) && (j==ymid_lo || j==ymid_hi);
                const Complex aaxis{is_on_axis ? ex : 0._rt, is_on_axis ? ei : 0._rt};

                return {            // Tuple contains:
                    aabssq,         // 0    max(|a|^2)
                    aabssq,         // 1    [|a|^2]
                    aabssq*x,       // 2    [|a|^2*x]
                    aabssq*x*x,     // 3    [|a|^2*x*x]
                    aabssq*y,       // 4    [|a|^2*y]
                    aabssq*y*y,     // 5    [|a|^2*y*y]
                    s1,             // 6    [jz], actually [fcK*sin(theta_j)*chi] for envelope
                    s2,             // 7    [rho], actually [fcK*cos(theta_j)*chi] for envelope
                    aaxis           // 8    axis(a)
                };
            });
    }

    auto [real_tup, cmpx_tup] = amrex::TupleSplit<m_insitu_nrp, m_insitu_ncp>(reduce_data.value());

    auto real_arr = amrex::tupleToArray(real_tup);

    for (int i=0; i<m_insitu_nrp; ++i) {
        if (i==0) {
            m_insitu_rdata[helmholtz_slice + i * nslices] = real_arr[i];
            m_insitu_sum_rdata[i] = std::max(m_insitu_sum_rdata[i], real_arr[i]);
        } else {
            m_insitu_rdata[helmholtz_slice + i * nslices] = real_arr[i]*dxdydz;
            m_insitu_sum_rdata[i] += real_arr[i]*dxdydz;
        }
    }

    auto cmpx_arr = amrex::tupleToArray(cmpx_tup);

    for (int i=0; i<m_insitu_ncp; ++i) {
        m_insitu_cdata[helmholtz_slice + i * nslices] = cmpx_arr[i] * mid_factor;
    }
}

void
Helmholtz::InSituWriteToFile (int step, amrex::Real time, bool is_last_step)
{
    if (!m_use_helmholtz) return;
    if (!m_insitu_period.doDiagnostics(step, time, is_last_step)) return;
    HIPACE_PROFILE("Helmholtz::InSituWriteToFile()");

#ifdef HIPACE_USE_OPENPMD
    // create subdirectory
    openPMD::auxiliary::create_directories(m_insitu_file_prefix);
#endif

    // zero pad the rank number;
    std::string::size_type n_zeros = 4;
    std::string rank_num = std::to_string(amrex::ParallelDescriptor::MyProc());
    std::string pad_rank_num = std::string(n_zeros-std::min(rank_num.size(), n_zeros),'0')+rank_num;

    // open file
    std::ofstream ofs{m_insitu_file_prefix + "/reduced_helmholtz." + pad_rank_num + ".txt",
        std::ofstream::out | std::ofstream::app | std::ofstream::binary};

    const int nslices_int = m_helmholtz_geom_3D.Domain().length(2);
    const std::size_t nslices = static_cast<std::size_t>(nslices_int);
    const int is_normalized_units = Hipace::m_normalized_units;

    // specify the structure of the data later available in python
    // avoid pointers to temporary objects as second argument, stack variables are ok
    const amrex::Vector<insitu_utils::DataNode> all_data{
        {"time"     , &time},
        {"step"     , &step},
        {"n_slices" , &nslices_int},
        {"z_lo"     , &m_helmholtz_geom_3D.ProbLo()[2]},
        {"z_hi"     , &m_helmholtz_geom_3D.ProbHi()[2]},
        {"is_normalized_units", &is_normalized_units},
        {"max(|a|^2)"     , &m_insitu_rdata[0], nslices},
        {"[|a|^2]"        , &m_insitu_rdata[1*nslices], nslices},
        {"[|a|^2*x]"      , &m_insitu_rdata[2*nslices], nslices},
        {"[|a|^2*x*x]"    , &m_insitu_rdata[3*nslices], nslices},
        {"[|a|^2*y]"      , &m_insitu_rdata[4*nslices], nslices},
        {"[|a|^2*y*y]"    , &m_insitu_rdata[5*nslices], nslices},
        {"[s1]"           , &m_insitu_rdata[6*nslices], nslices},
        {"[s2]"           , &m_insitu_rdata[7*nslices], nslices},
        {"axis(a)"        , &m_insitu_cdata[0], nslices},
        {"integrated", {
            {"max(|a|^2)"     , &m_insitu_sum_rdata[0]},
            {"[|a|^2]"        , &m_insitu_sum_rdata[1]},
            {"[|a|^2*x]"      , &m_insitu_sum_rdata[2]},
            {"[|a|^2*x*x]"    , &m_insitu_sum_rdata[3]},
            {"[|a|^2*y]"      , &m_insitu_sum_rdata[4]},
            {"[|a|^2*y*y]"    , &m_insitu_sum_rdata[5]},
            {"[s1]"           , &m_insitu_sum_rdata[6]},
            {"[s2]"           , &m_insitu_sum_rdata[7]}
        }}
    };

    if (ofs.tellp() == 0) {
        // write JSON header containing a NumPy structured datatype
        insitu_utils::write_header(all_data, ofs);
    }

    // write binary data according to datatype in header
    insitu_utils::write_data(all_data, ofs);

    // close file
    ofs.close();
    // assert no file errors
#ifdef HIPACE_USE_OPENPMD
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu helmholtz diagnostics");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu helmholtz diagnostics. "
        "Maybe the specified subdirectory does not exist");
#endif

    // reset arrays for insitu data
    for (auto& x : m_insitu_rdata) x = 0.;
    for (auto& x : m_insitu_sum_rdata) x = 0.;
    for (auto& x : m_insitu_cdata) x = 0.;
}
