/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: Andrew Myers, MaxThevenet, Remi Lehe, Severin Diederichs
 * Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_ParticleUtil_H_
#define HIPACE_ParticleUtil_H_

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_Random.H>

/** \brief Basic helper functions that can be used for both plasma and beam species */
namespace ParticleUtil
{
    using namespace amrex::literals;

    /** During particles initialization, we loop over nppc particles in 1 cell.
     * This function returns the position, in cell size unit, of particle number i_part
     * so that particles are distributed uniformly within the cell.
     *
     * \param[in,out] r position of the particle, modified by the function
     * \param[in] nppc number of particles per cell in each direction
     * \param[in] i_part index of current particle, between 0 and nppc-1
     * \param[in] engine random engine
     * \param[in] rand whether to randomize position in each cell, per dimension
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void get_position_unit_cell (amrex::Real* r, const amrex::IntVect& nppc, int i_part, const amrex::RandomEngine& engine, const amrex::GpuArray<int, 3> rand)
    {
        int nx = nppc[0];
        int ny = nppc[1];
        int nz = nppc[2];

        int ix_part = i_part/(ny * nz);
        int iy_part = (i_part % (ny * nz)) % ny;
        int iz_part = (i_part % (ny * nz)) / ny;

        r[0] = rand[0] ? amrex::Random(engine) : (0.5_rt + ix_part)/nx;
        r[1] = rand[1] ? amrex::Random(engine) : (0.5_rt + iy_part)/ny;
        r[2] = rand[2] ? amrex::Random(engine) : (0.5_rt + iz_part)/nz;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void get_position_unit_cell (amrex::Real* r, const amrex::IntVect& nppc, int i_part)
    {
        int nx = nppc[0];
        int ny = nppc[1];
        int nz = nppc[2];

        int ix_part = i_part/(ny * nz);
        int iy_part = (i_part % (ny * nz)) % ny;
        int iz_part = (i_part % (ny * nz)) / ny;

        r[0] = (0.5_rt + ix_part)/nx;
        r[1] = (0.5_rt + iy_part)/ny;
        r[2] = (0.5_rt + iz_part)/nz;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void get_position_unit_cell_fine (amrex::Real* r, bool& do_init, int i_part,
                                      const amrex::IntVect& nppc_coarse,
                                      const amrex::IntVect& nppc_fine,
                                      const int n_fine_transition_cells,
                                      const int i_fine_transition_cells)
    {
        if (i_fine_transition_cells == 0) {
            int nx = nppc_coarse[0];
            int ny = nppc_coarse[1];
            if (i_part < nx * ny) {
                do_init = true;
                int ix_part = i_part % nx;
                int iy_part = i_part / nx;
                r[0] = (0.5_rt + ix_part)/nx;
                r[1] = (0.5_rt + iy_part)/ny;
            } else {
                do_init = false;
            }
        } else {
            do_init = true;
            int nx_fine = nppc_fine[0];
            int ny_fine = nppc_fine[1];
            int ix_part_fine = i_part % nx_fine;
            int iy_part_fine = i_part / nx_fine;

            int nx_coarse = nppc_coarse[0];
            int ny_coarse = nppc_coarse[1];
            int ix_part_coarse = (ix_part_fine * nx_coarse) / nx_fine;
            int iy_part_coarse = (iy_part_fine * ny_coarse) / ny_fine;

            amrex::Real s = amrex::Real(i_fine_transition_cells) / (n_fine_transition_cells + 1);
            s = 1.5_rt*s - 0.5_rt*s*s*s;

            r[0] = ((0.5_rt + ix_part_coarse) / nx_coarse) * (1._rt - s) +
                   ((0.5_rt + ix_part_fine) / nx_fine) * s;
            r[1] = ((0.5_rt + iy_part_coarse) / ny_coarse) * (1._rt - s) +
                   ((0.5_rt + iy_part_fine) / ny_fine) * s;
        }
    }

    /** Return momentum of 1 particle from a Gaussian random draw.
     * \param[in,out] u 3D momentum of 1 particle, modified by this function
     * \param[in] u_mean Mean value of the random distribution in each dimension
     * \param[in] u_std standard deviation of the random distribution in each dimension
     * \param[in] engine random number engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void get_gaussian_random_momentum (amrex::Real* u, const amrex::RealVect u_mean,
                                       const amrex::RealVect u_std,
                                       const amrex::RandomEngine& engine)
    {
        amrex::Real ux_th = amrex::RandomNormal(0.0, u_std[0], engine);
        amrex::Real uy_th = amrex::RandomNormal(0.0, u_std[1], engine);
        amrex::Real uz_th = amrex::RandomNormal(0.0, u_std[2], engine);

        u[0] = u_mean[0] + ux_th;
        u[1] = u_mean[1] + uy_th;
        u[2] = u_mean[2] + uz_th;
    }
}

#endif
