#include "Laser.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

amrex::IntVect Laser::m_slices_nguards = {-1, -1, -1};

void
Laser::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    queryWithParser(pp, "a0", m_a0);
    queryWithParser(pp, "w0", m_w0);
    queryWithParser(pp, "L0", m_L0);
    queryWithParser(pp, "lambda0", m_lambda0);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    queryWithParser(pp, "position_mean", loc_array); // could potentially be getWithParser
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
    queryWithParser(pp, "position_std", loc_array); // could potentially be getWithParser
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_std[idim] = loc_array[idim];

}


void
Laser::InitData (const amrex::BoxArray& slice_ba,
                 const amrex::DistributionMapping& slice_dm)
{
    HIPACE_PROFILE("Laser::InitData()");
    // Alloc 2D slices
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = std::max(1, Hipace::m_depos_order_xy);
    m_slices_nguards = {nguards_xy, nguards_xy, 0};

    for (int islice=0; islice<WhichLaserSlice::N; islice++) {
        m_slices[islice].define(
            slice_ba, slice_dm, 1, m_slices_nguards, // prev Comps[islice]["N"] instead of 1
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[islice].setVal(0.0);
    }
}


void
Laser::PrepareLaserSlice (const amrex::Geometry& geom, const int islice)
{
    HIPACE_PROFILE("Laser::PrepareLaserSlice()");
    using namespace amrex::literals;

    const amrex::Real a0 = m_a0;

    const auto plo = geom.ProbLoArray();
    const amrex::Real* dx = geom.CellSize();
    amrex::IntVect lo = {0, 0, 0};

    const amrex::GpuArray<amrex::Real, 3> pos_mean = {m_position_mean[0], m_position_mean[1],
                                                      m_position_mean[2]};
    const amrex::GpuArray<amrex::Real, 3> pos_std = {m_position_std[0], m_position_std[1],
                                                     m_position_std[2]};
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    const amrex::Real z = plo[2] + (islice+0.5_rt)*dx_arr[2];
    const amrex::Real delta_z = (z - pos_mean[2]) / pos_std[2];
    const amrex::Real long_pos_factor =  std::exp( -(delta_z*delta_z) );

    amrex::MultiFab& slice_this    = getSlices(WhichLaserSlice::This);
    amrex::MultiFab& slice_AbsSq   = getSlices(WhichLaserSlice::AbsSq);
    amrex::MultiFab& slice_AbsSqDx = getSlices(WhichLaserSlice::AbsSqDx);
    amrex::MultiFab& slice_AbsSqDy = getSlices(WhichLaserSlice::AbsSqDy);

    const int dcomp = 0; // NOTE, this may not always be true, to be checked
    const int scomp = 0; // NOTE, this may not always be true, to be checked


#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(slice_this, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const & array_this = slice_this.array(mfi);
        amrex::Array4<amrex::Real> const & array_AbsSq = slice_AbsSq.array(mfi);
        amrex::Array4<amrex::Real> const & array_AbsSqDx = slice_AbsSqDx.array(mfi);
        amrex::Array4<amrex::Real> const & array_AbsSqDy = slice_AbsSqDy.array(mfi);

        // setting this Laser slice to the initial slice (TO BE REPLACED BY COPY FROM 3D ARRAY)
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const amrex::Real x = (i+0.5)*dx_arr[0]+plo[0];
                const amrex::Real y = (j+0.5)*dx_arr[1]+plo[1];

                const amrex::Real delta_x = (x - pos_mean[0]) / pos_std[0];
                const amrex::Real delta_y = (y - pos_mean[1]) / pos_std[1];
                const amrex::Real trans_pos_factor =  std::exp( -(delta_x*delta_x
                                                                        + delta_y*delta_y) );

                array_this(i,j,k,dcomp) =  a0*trans_pos_factor*long_pos_factor;

            }
            );

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                array_AbsSq(i,j,k,dcomp) = std::abs( array_this(i,j,k,dcomp)
                                                    *array_this(i,j,k,dcomp));
            }
            );

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* finite difference along x */
                array_AbsSqDx(i,j,k,dcomp) = 1._rt / (2.0_rt*dx_arr[0]) *
                                         (array_AbsSq(i+1+lo[0], j+lo[1], k, scomp)
                                          - array_AbsSq(i-1+lo[0], j+lo[1], k, scomp));
            }
            );

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* finite difference along y */
                array_AbsSqDy(i,j,k,dcomp) = 1._rt  / (2.0_rt*dx_arr[1]) *
                                         (array_AbsSq(i+lo[0], j+1+lo[1], k, scomp)
                                         - array_AbsSq(i+lo[0], j-1+lo[1], k, scomp));
            }
            );
    }

}
