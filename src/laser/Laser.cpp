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
    queryWithParser(pp, "lambda0", m_lambda0);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    queryWithParser(pp, "position_mean", loc_array);
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
        m_slices[islice].define(
            slice_ba, slice_dm, 1, m_slices_nguards, /* prev Comps[islice]["N"] instead of 1 */
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[islice].setVal(0.0);
    }
}

void
Laser::Init3DEnvelope (int step, amrex::Box bx, const amrex::Geometry& gm)
{
    if (!m_use_laser) return;
    HIPACE_PROFILE("Laser::Init3DEnvelope()");
    // bx.grow(m_slices_nguards);
    // Allocate the 3D field on this box
    // m_F.resize(bx, m_nfields_3d, amrex::The_Pinned_Arena(), m_slices_nguards);
    m_F.resize(bx, m_nfields_3d, amrex::The_Pinned_Arena());
    amrex::AllPrint()<<"rank "<<amrex::ParallelDescriptor::MyProc()<<" "<<bx<<'\n';

    if (step > 0) return;

    // Loop over slices
    for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
        // Compute initial field on the current (device) slice
        PrepareLaserSlice(gm, isl);
        // Copy (device) slice to (host) 3D array
        Copy(isl, true);
    }
    amrex::AllPrint()<<"rank "<<amrex::ParallelDescriptor::MyProc()<<": init, "<<m_F.max()<<' '<<m_F.min()<<'\n';
}

void
Laser::Copy (int isl, bool to3d)
{
    amrex::MultiFab& this_slice = m_slices[WhichLaserSlice::This];
    amrex::MultiFab& newt_slice = m_slices[WhichLaserSlice::NextTime];
    amrex::MultiFab& oldt_slice = m_slices[WhichLaserSlice::PrevTime1];
    amrex::MultiFab& zeta_slice = m_slices[WhichLaserSlice::PrevZeta1];

    // for ( amrex::MFIter mfi(this_slice, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
    for ( amrex::MFIter mfi(this_slice, false); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> this_arr = this_slice.array(mfi);
        amrex::Array4<amrex::Real> newt_arr = newt_slice.array(mfi);
        amrex::Array4<amrex::Real> zeta_arr = zeta_slice.array(mfi);
        amrex::Array4<amrex::Real> oldt_arr = oldt_slice.array(mfi);
        amrex::Array4<amrex::Real> host_arr = m_F.array();
        amrex::ParallelFor(
        bx, 1,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
        {
            // n should always be 0 here
            if (to3d){
                // next time slice into new host
                host_arr(i,j,isl,n) = newt_arr(i,j,k,n);
                // this slice into old host
                host_arr(i,j,isl,n+2) = this_arr(i,j,k,n);
                // this slice into previous zeta slice
                zeta_arr(i,j,k,n) = this_arr(i,j,k,n);
            } else {
                // Get current slice from 3D host array, both current and previous time step
                this_arr(i,j,k,n) = host_arr(i,j,isl,n);
                oldt_arr(i,j,k,n) = host_arr(i,j,isl,n+2);
            }
        });
    }
}

void
Laser::AdvanceSlice(const Fields& fields)
{
    using namespace amrex::literals;
    amrex::MultiFab& this_slice = m_slices[WhichLaserSlice::This];
    amrex::MultiFab& newt_slice = m_slices[WhichLaserSlice::NextTime];
    amrex::MultiFab& oldt_slice = m_slices[WhichLaserSlice::PrevTime1];
    amrex::MultiFab& zeta_slice = m_slices[WhichLaserSlice::PrevZeta1];

    for ( amrex::MFIter mfi(this_slice, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> this_arr = this_slice.array(mfi);
        amrex::Array4<amrex::Real> newt_arr = newt_slice.array(mfi);
        amrex::Array4<amrex::Real> zeta_arr = zeta_slice.array(mfi);
        amrex::Array4<amrex::Real> oldt_arr = oldt_slice.array(mfi);
        amrex::ParallelFor(
        bx, 1,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
        {
            newt_arr(i,j,k,n) = .9_rt * this_arr(i,j,k,n);
        });
    }
}

void
Laser::PrepareLaserSlice (const amrex::Geometry& geom, const int islice)
{
    if (!m_use_laser) return;
    HIPACE_PROFILE("Laser::PrepareLaserSlice()");
    using namespace amrex::literals;

    const amrex::Real a0 = m_a0;

    const auto plo = geom.ProbLoArray();
    amrex::Real const * const dx = geom.CellSize();

    const amrex::GpuArray<amrex::Real, 3> pos_mean = {m_position_mean[0], m_position_mean[1],
                                                      m_position_mean[2]};
    const amrex::GpuArray<amrex::Real, 3> pos_size = {m_w0[0], m_w0[1], m_L0};
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    const amrex::Real z = plo[2] + (islice+0.5_rt)*dx_arr[2];
    // amrex::AllPrint()<<"rank "<<amrex::ParallelDescriptor::MyProc()<<" plo "<<plo[2]<<" z "<<z<<'\n';
    const amrex::Real delta_z = (z - pos_mean[2]) / pos_size[2];
    const amrex::Real long_pos_factor =  std::exp( -(delta_z*delta_z) );

    amrex::MultiFab& slice_this = getSlices(WhichLaserSlice::NextTime);

    const int dcomp = 0; // NOTE, this may change when we use slices with Comps

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // for ( amrex::MFIter mfi(slice_this, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
    for ( amrex::MFIter mfi(slice_this, false); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        // amrex::AllPrint()<<"here bx "<<bx<<'\n';
        amrex::Array4<amrex::Real> const & array_this = slice_this.array(mfi);

        // setting this Laser slice to the initial slice (TO BE REPLACED BY COPY FROM 3D ARRAY)
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

                array_this(i,j,k,dcomp) =  a0*trans_pos_factor*long_pos_factor;

            }
            );
    }

}
