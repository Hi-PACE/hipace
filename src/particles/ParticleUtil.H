#ifndef HIPACE_ParticleUtil_H_
#define HIPACE_ParticleUtil_H_

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>

namespace ParticleUtil
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void get_position_unit_cell (amrex::Real* r, const amrex::IntVect& nppc, int i_part)
    {
        int nx = nppc[0];
        int ny = nppc[1];
        int nz = nppc[2];

        int ix_part = i_part/(ny * nz);
        int iy_part = (i_part % (ny * nz)) % ny;
        int iz_part = (i_part % (ny * nz)) / ny;

        r[0] = (0.5+ix_part)/nx;
        r[1] = (0.5+iy_part)/ny;
        r[2] = (0.5+iz_part)/nz;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void get_gaussian_random_momentum (amrex::Real* u, const amrex::Vector< amrex::Real>& u_mean, const amrex::Vector< amrex::Real>& u_std)
    {
        amrex::Real ux_th = amrex::RandomNormal(0.0, u_std[0]);
        amrex::Real uy_th = amrex::RandomNormal(0.0, u_std[1]);
        amrex::Real uz_th = amrex::RandomNormal(0.0, u_std[2]);

        u[0] = u_mean[0] + ux_th;
        u[1] = u_mean[1] + uy_th;
        u[2] = u_mean[2] + uz_th;
    }
}

#endif
