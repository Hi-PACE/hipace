#include "GridCurrent.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

GridCurrent::GridCurrent ()
{
    amrex::ParmParse ppa("grid_current");

    ppa.query("use_grid_current", m_use_grid_current);
    ppa.query("amplitude", m_amplitude);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    if (ppa.query("position_mean", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
    }
    if (ppa.query("position_std", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_std[idim] = loc_array[idim];
    }

    // ppa.query("do_adaptive_time_step", m_do_adaptive_time_step);
    // ppa.query("nt_per_omega_betatron", m_nt_per_omega_betatron);
}

void
GridCurrent::DepositCurrentSlice (Fields& fields, const amrex::Geometry& geom, int const lev, const int islice)
{
    HIPACE_PROFILE("GridCurrent::DepositCurrentSlice()");
    using namespace amrex::literals;

    if (m_use_grid_current == 0) return;

    const auto plo = geom.ProbLoArray();
    amrex::Real const * AMREX_RESTRICT dx = geom.CellSize();

    const amrex::GpuArray<amrex::Real, 3>  a_pos_mean = {m_position_mean[0], m_position_mean[1], m_position_mean[2]};
    const amrex::GpuArray<amrex::Real, 3>  a_pos_std = {m_position_std[0], m_position_std[1], m_position_std[2]};

    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
    // const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

    // Extract the fields currents
    amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
    amrex::MultiFab jz(S, amrex::make_alias, Comps[WhichSlice::This]["jz"], 1);
    jz.setVal(0.);
    // Extract FabArray for this box
    amrex::FArrayBox& jz_fab = jz[0];
    const amrex::Real loc_amplitude = m_amplitude;
    // const int comp_index = Comps[WhichSlice::This]["jz"];

    const amrex::Real z = plo[2] + islice*dx_arr[2];
    const amrex::Real delta_z = (z - a_pos_mean[2]) / a_pos_std[2];
    const amrex::Real long_pos_factor =  exp( -0.5_rt*(delta_z*delta_z) );

    for ( amrex::MFIter mfi(S, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
        // amrex::Array4<amrex::Real> const& jz_arr = S.array(mfi);
        // amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                amrex::Real x = plo[0] + i*dx_arr[0];
                amrex::Real y = plo[1] + j*dx_arr[1];

                amrex::Real delta_x = (x - a_pos_mean[0]) / a_pos_std[0];
                amrex::Real delta_y = (y - a_pos_mean[1]) / a_pos_std[1];
                const amrex::Real trans_pos_factor =  exp( -0.5_rt*(delta_x*delta_x + delta_y*delta_y) );

                jz_arr(i, j, k) += -loc_amplitude*trans_pos_factor*long_pos_factor;
            }
            );
    }

    // z(slice);
    // for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
    //     const amrex::Box& bx = mfi.tilebox();
    //     amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
    //     amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
    //     amrex::ParallelFor(
    //         bx,
    //         [=] AMREX_GPU_DEVICE(int i, int j, int k)
    //         {
    //             // (k ignored)
    //         });
    // }
}
