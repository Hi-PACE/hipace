/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "GridCurrent.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "GPUUtil.H"
#include "Constants.H"

GridCurrent::GridCurrent ()
{
    amrex::ParmParse pp("grid_current");

    if (queryWithParser(pp, "use_grid_current", m_use_grid_current) ) {
        getWithParser(pp, "peak_current_density", m_peak_current_density);
        getWithParser(pp, "position_mean", m_position_mean);
        getWithParser(pp, "position_std", m_position_std);
    }
}

void
GridCurrent::DepositCurrentSlice (Fields& fields, const amrex::Geometry& geom, int const lev,
                                  const int islice)
{
    if (m_use_grid_current == 0) return;

    HIPACE_PROFILE("GridCurrent::DepositCurrentSlice()");
    using namespace amrex::literals;

    const auto plo = geom.ProbLoArray();
    amrex::Real const * AMREX_RESTRICT dx = geom.CellSize();

    const amrex::GpuArray<amrex::Real, 3> pos_mean = {m_position_mean[0], m_position_mean[1],
                                                      m_position_mean[2]};
    const amrex::GpuArray<amrex::Real, 3> pos_std = {m_position_std[0], m_position_std[1],
                                                     m_position_std[2]};
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    // Extract the longitudinal beam current
    amrex::MultiFab& S = fields.getSlices(lev);

    const amrex::Real z = plo[2] + islice*dx_arr[2];
    const amrex::Real delta_z = (z - pos_mean[2]) / pos_std[2];
    const amrex::Real long_pos_factor =  std::exp( -0.5_rt*(delta_z*delta_z) );
    const amrex::Real loc_peak_current_density = m_peak_current_density;

    for ( amrex::MFIter mfi(S, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        Array2<amrex::Real> const jz_arr = S.array(mfi, Hipace::m_explicit ?
            Comps[WhichSlice::This]["jz_beam"] : Comps[WhichSlice::This]["jz"]);

        amrex::ParallelFor( to2D(bx),
        [=] AMREX_GPU_DEVICE(int i, int j)
        {
            const amrex::Real x = plo[0] + (i+0.5_rt)*dx_arr[0];
            const amrex::Real y = plo[1] + (j+0.5_rt)*dx_arr[1];

            const amrex::Real delta_x = (x - pos_mean[0]) / pos_std[0];
            const amrex::Real delta_y = (y - pos_mean[1]) / pos_std[1];
            const amrex::Real trans_pos_factor =  std::exp( -0.5_rt*(delta_x*delta_x
                                                                    + delta_y*delta_y) );

            jz_arr(i, j) += loc_peak_current_density*trans_pos_factor*long_pos_factor;
        });
    }
}
