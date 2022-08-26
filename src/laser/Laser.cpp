#include "Laser.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

void
Laser::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    m_use_laser = queryWithParser(pp, "a0", m_a0);
    if (!m_use_laser) return;
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
    queryWithParser(pp, "position_mean", loc_array);
    queryWithParser(pp, "3d_on_host", m_3d_on_host);
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
}

void
Laser::Init3DEnvelope (int step, amrex::Box bx, const amrex::Geometry& gm)
{
    if (!m_use_laser) return;
    HIPACE_PROFILE("Laser::Init3DEnvelope()");
    // Allocate the 3D field on this box
    m_F.resize(bx, m_nfields_3d, m_3d_on_host ? amrex::The_Pinned_Arena() : amrex::The_Arena());

    if (step > 0) return;

    // Loop over slices
    for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
        // Compute initial field on the current (device) slice
        InitLaserSlice(gm, isl);
        // Copy: (device) slice to (host) 3D array
        Copy(isl, true, true);
    }
}

void
Laser::Copy (int isl, bool to3d, bool init)
{
    HIPACE_PROFILE("Laser::Copy()");
    amrex::MultiFab& nm1jm1 = m_slices[WhichLaserSlice::nm1jm1];
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

    for ( amrex::MFIter mfi(n00j00, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> nm1jm1_arr = nm1jm1.array(mfi);
        amrex::Array4<amrex::Real> nm1j00_arr = nm1j00.array(mfi);
        amrex::Array4<amrex::Real> nm1jp1_arr = nm1jp1.array(mfi);
        amrex::Array4<amrex::Real> nm1jp2_arr = nm1jp2.array(mfi);
        amrex::Array4<amrex::Real> n00j00_arr = n00j00.array(mfi);
        amrex::Array4<amrex::Real> n00jp1_arr = n00jp1.array(mfi);
        amrex::Array4<amrex::Real> n00jp2_arr = n00jp2.array(mfi);
        amrex::Array4<amrex::Real> np1j00_arr = np1j00.array(mfi);
        amrex::Array4<amrex::Real> np1jp1_arr = np1jp1.array(mfi);
        amrex::Array4<amrex::Real> np1jp2_arr = np1jp2.array(mfi);
        amrex::Array4<amrex::Real> host_arr = m_F.array();
        amrex::ParallelFor(
        bx, 2,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
        {
            // n should always be 0 here
            if (to3d){
                // next time slice into new host
                // host_arr(i,j,isl,n) = newt_arr(i,j,k,n);
                host_arr(i,j,isl,n) = np1j00_arr(i,j,k,n);
                if (init) host_arr(i,j,isl,n+2) = np1j00_arr(i,j,k,n);
                // this slice into old host
                // Cannot update slice isl of old array because we'll need the
                // previous value when computing the next slice.
                if (isl+1 <= izmax) host_arr(i,j,isl+1,n+2) = n00jp1_arr(i,j,k,n);
                if (isl+2 <= izmax) host_arr(i,j,isl+2,n+2) = n00jp2_arr(i,j,k,n);
            } else {
                if (isl-1 >= izmin) nm1jm1_arr(i,j,k,n) = host_arr(i,j,isl-1,n+2);
                nm1j00_arr(i,j,k,n) = host_arr(i,j,isl,n+2);
                if (isl+1 <= izmax) nm1jp1_arr(i,j,k,n) = host_arr(i,j,isl+1,n+2);
                if (isl+2 <= izmax) nm1jp2_arr(i,j,k,n) = host_arr(i,j,isl+2,n+2);
                n00jp2_arr(i,j,k,n) = n00jp1_arr(i,j,k,n);
                n00jp1_arr(i,j,k,n) = n00j00_arr(i,j,k,n);
                n00j00_arr(i,j,k,n) = host_arr(i,j,isl,n);
                np1jp2_arr(i,j,k,n) = np1jp1_arr(i,j,k,n);
                np1jp1_arr(i,j,k,n) = np1j00_arr(i,j,k,n);
            }
        });
    }
}

void
Laser::AdvanceSlice(const Fields& fields, const amrex::Real* cellsize, const amrex::Real dt)
{
    using namespace amrex::literals;
    const amrex::Real dx = cellsize[0];
    const amrex::Real dy = cellsize[1];
    const amrex::Real dz = cellsize[2];
    constexpr amrex::Real pi = MathConst::pi;

    const amrex::Real c = get_phys_const().c;
    const amrex::Real k0 = 2.*MathConst::pi/m_lambda0;

    amrex::MultiFab& nm1jm1 = m_slices[WhichLaserSlice::nm1jm1];
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

    for ( amrex::MFIter mfi(n00j00, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        acoeff_imag.resize(bx, 1, amrex::The_Arena());
        rhs.resize(bx, 2, amrex::The_Arena());
        amrex::Array4<amrex::Real> nm1jm1_arr = nm1jm1.array(mfi);
        amrex::Array4<amrex::Real> nm1j00_arr = nm1j00.array(mfi);
        amrex::Array4<amrex::Real> nm1jp1_arr = nm1jp1.array(mfi);
        amrex::Array4<amrex::Real> nm1jp2_arr = nm1jp2.array(mfi);
        amrex::Array4<amrex::Real> n00j00_arr = n00j00.array(mfi);
        amrex::Array4<amrex::Real> n00jp1_arr = n00jp1.array(mfi);
        amrex::Array4<amrex::Real> n00jp2_arr = n00jp2.array(mfi);
        amrex::Array4<amrex::Real> np1j00_arr = np1j00.array(mfi);
        amrex::Array4<amrex::Real> np1jp1_arr = np1jp1.array(mfi);
        amrex::Array4<amrex::Real> np1jp2_arr = np1jp2.array(mfi);
        amrex::Array4<amrex::Real> rhs_arr    = rhs.array();
        amrex::Array4<amrex::Real> acoeff_imag_arr = acoeff_imag.array();
        amrex::ParallelFor(
        bx, 1,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
        {
            /* First, define a few constants and temporary variables */
            // Calculate complex arguments (theta) needed
            const amrex::Real tj00 = std::atan2(n00j00_arr(i,j,k,1), n00j00_arr(i,j,k,0));
            const amrex::Real tjp1 = std::atan2(n00jp1_arr(i,j,k,1), n00jp1_arr(i,j,k,0));
            const amrex::Real tjp2 = std::atan2(n00jp2_arr(i,j,k,1), n00jp2_arr(i,j,k,0));
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
            amrex::Real lapR =
                (nm1j00_arr(i+1,j,k,0)+nm1j00_arr(i-1,j,k,0)-2._rt*nm1j00_arr(i,j,k,0))/(dx*dx) +
                (nm1j00_arr(i,j+1,k,0)+nm1j00_arr(i,j-1,k,0)-2._rt*nm1j00_arr(i,j,k,0))/(dy*dy);
            amrex::Real lapI =
                (nm1j00_arr(i+1,j,k,1)+nm1j00_arr(i-1,j,k,1)-2._rt*nm1j00_arr(i,j,k,1))/(dx*dx) +
                (nm1j00_arr(i,j+1,k,1)+nm1j00_arr(i,j-1,k,1)-2._rt*nm1j00_arr(i,j,k,1))/(dy*dy);
            // D_j^n as defined in Benedetti's 2017 paper
            amrex::Real djn = ( 3._rt*tj00 - 4._rt*tjp1 + tjp2 ) / (2._rt*dz);

            /* Then, calculate the few arrays required */
            // Imag acoeff term (the Real part is just a scalar defined above)
            acoeff_imag_arr(i,j,k,0) = -2._rt*( k0 + djn ) / (c*dt);
            // Real RHS term
            rhs_arr(i,j,k,0) =
                + 4._rt/(c*dt*dz)*((np1jp1_arr(i,j,k,0)-nm1jp1_arr(i,j,k,0))*cdt1 -
                                   (np1jp1_arr(i,j,k,1)-nm1jp1_arr(i,j,k,1))*sdt1)
                - 1._rt/(c*dt*dz)*((np1jp2_arr(i,j,k,0)-nm1jp2_arr(i,j,k,0))*cdt2 -
                                   (np1jp2_arr(i,j,k,1)-nm1jp2_arr(i,j,k,1))*sdt2)
                - 4._rt/(c*c*dt*dt)*n00j00_arr(i,j,k,0)
                + 0.0000000
                - lapR
                + 3._rt/(c*dt*dz)  * nm1j00_arr(i,j,k,0)
                - 2._rt/(c*dt)*djn * nm1j00_arr(i,j,k,1)
                + 2._rt/(c*c*dt*dt) * nm1j00_arr(i,j,k,0);

            // Imag RHS term
            rhs_arr(i,j,k,1) =
                + 4._rt/(c*dt*dz)*((np1jp1_arr(i,j,k,1)-nm1jp1_arr(i,j,k,1))*cdt1 +
                                   (np1jp1_arr(i,j,k,0)-nm1jp1_arr(i,j,k,0))*sdt1)
                - 1._rt/(c*dt*dz)*((np1jp2_arr(i,j,k,1)-nm1jp2_arr(i,j,k,1))*cdt2 +
                                   (np1jp2_arr(i,j,k,0)-nm1jp2_arr(i,j,k,0))*sdt2)
                - 4._rt/(c*c*dt*dt)*n00j00_arr(i,j,k,1)
                - 0.0000000
                - lapI
                + 3._rt/(c*dt*dz)  * nm1j00_arr(i,j,k,1)
                + 2._rt/(c*dt)*djn * nm1j00_arr(i,j,k,0)
                + 2._rt/(c*c*dt*dt) * nm1j00_arr(i,j,k,1);
        });
    }
/*
    // construct slice geometry
    // Set the lo and hi of domain and probdomain in the z direction
    amrex::RealBox tmp_probdom({AMREX_D_DECL(Geom(lev).ProbLo(Direction::x),
                                             Geom(lev).ProbLo(Direction::y),
                                             Geom(lev).ProbLo(Direction::z))},
                               {AMREX_D_DECL(Geom(lev).ProbHi(Direction::x),
                                             Geom(lev).ProbHi(Direction::y),
                                             Geom(lev).ProbHi(Direction::z))});
    amrex::Box tmp_dom = Geom(lev).Domain();
    const amrex::Real hi = Geom(lev).ProbHi(Direction::z);
    const amrex::Real lo = hi - dz;
    tmp_probdom.setLo(Direction::z, lo);
    tmp_probdom.setHi(Direction::z, hi);
    tmp_dom.setSmall(Direction::z, 0);
    tmp_dom.setBig(Direction::z, 0);
    amrex::Geometry slice_geom = amrex::Geometry(
        tmp_dom, tmp_probdom, Geom(lev).Coord(), Geom(lev).isPeriodic());

    slice_geom.setPeriodicity({0,0,0});
    if (!m_mg) {
        m_mg = std::make_unique<hpmg::MultiGrid>(slice_geom);
    }
    const int max_iters = 200;
    // [0] means we're taking the first (and only) box in these MF
    // Mult == acoeff
    // S == rhs
    m_hpmg->solve2(np1j00[0], S[0], Mult[0], m_MG_tolerance_rel, m_MG_tolerance_abs,
    max_iters, m_MG_verbose);
*/
}

void
Laser::InitLaserSlice (const amrex::Geometry& geom, const int islice)
{
    if (!m_use_laser) return;
    HIPACE_PROFILE("Laser::InitLaserSlice()");
    using namespace amrex::literals;

    const amrex::Real a0 = m_a0;

    const auto plo = geom.ProbLoArray();
    amrex::Real const * const dx = geom.CellSize();

    const amrex::GpuArray<amrex::Real, 3> pos_mean = {m_position_mean[0], m_position_mean[1],
                                                      m_position_mean[2]};
    const amrex::GpuArray<amrex::Real, 3> pos_size = {m_w0[0], m_w0[1], m_L0};
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    const amrex::Real z = plo[2] + (islice+0.5_rt)*dx_arr[2];
    const amrex::Real delta_z = (z - pos_mean[2]) / pos_size[2];
    const amrex::Real long_pos_factor =  std::exp( -(delta_z*delta_z) );

    amrex::MultiFab& slice_this = getSlices(WhichLaserSlice::np1j00);

    const int dcomp = 0; // NOTE, this may change when we use slices with Comps

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(slice_this, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const & array_this = slice_this.array(mfi);

        // Initialize a Gaussian laser envelope on slice islice
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const amrex::Real x = (i+0.5_rt)*dx_arr[0]+plo[0];
                const amrex::Real y = (j+0.5_rt)*dx_arr[1]+plo[1];

                const amrex::Real delta_x = (x - pos_mean[0]) / pos_size[0];
                const amrex::Real delta_y = (y - pos_mean[1]) / pos_size[1];
                const amrex::Real trans_pos_factor =  std::exp( -(delta_x*delta_x
                                                                        + delta_y*delta_y) );
                // Real part
                array_this(i,j,k,dcomp  ) =  a0*trans_pos_factor*long_pos_factor;
                // Imaginary part
                array_this(i,j,k,dcomp+1) =  0._rt;
            }
            );
    }

}
