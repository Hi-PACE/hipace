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
     */
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
