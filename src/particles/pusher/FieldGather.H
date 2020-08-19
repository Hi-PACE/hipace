/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef FIELDGATHER_H_
#define FIELDGATHER_H_

// #include "Particles/Pusher/GetAndSetPosition.H"
#include "../ShapeFactors.H"
// #include "Utils/WarpX_Complex.H"

#include <AMReX.H>

/**
 * \brief Field gather for a single particle
 *
 * \param xp, yp, zp                : Particle position coordinates
 * \param ExmByp, EypBxp, Ezp             : Electric field on particles.
 * \param Bxp, Byp, Bzp             : Magnetic field on particles.
 * \param exmby_arr eypbx_arr ez_arr      : Array4 of the electric field, either full array or tile.
 * \param bx_arr by_arr bz_arr      : Array4 of the magnetic field, either full array or tile.
 * \param ex_type, ey_type, ez_type : IndexType of the electric field
 * \param bx_type, by_type, bz_type : IndexType of the magnetic field
 * \param dx                        : 3D cell spacing
 * \param xyzmin                    : Physical lower bounds of domain in x, y, z.
 * \param lo                        : Index lower bounds of domain.
 * \param n_rz_azimuthal_modes      : Number of azimuthal modes when using RZ geometry
 */
template <int depos_order_xy, int depos_order_z>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& ExmByp,
                     amrex::ParticleReal& EypBxp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& exmby_arr,
                     amrex::Array4<amrex::Real const> const& eypbx_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::GpuArray<amrex::Real, 3>& dx,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin,
                     const amrex::Dim3& lo)
{
    using namespace amrex;

    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];

    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    const amrex::Real zmin = xyzmin[2];

    // x,y,z direction
    const amrex::Real x = (xp-xmin)*dxi;
    const amrex::Real y = (yp-ymin)*dyi;
    const amrex::Real z = (zp-zmin)*dzi;


    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, x - 0.5_rt);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, y - 0.5_rt);

    // // z directioni;
    amrex::Real sz_cell[depos_order_z + 1];
    const int l_cell = compute_shape_factor<depos_order_z>(sz_cell, z - 0.5_rt);

    // std::cout << "l_cell " << l_cell << "\n";
    // std::cout << "lo.z " << lo.z << "\n";
    // std::cout << "lo.x " << lo.x << "\n";
    // std::cout << "lo.y " << lo.y << "\n";
    // std::cout << "depos_order_z " << depos_order_z << "\n";
    // std::cout << "depos_order_xy " << depos_order_xy << "\n";

    // Gather field on particle ExmByp from field on grid exmby_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                ExmByp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    exmby_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz);
                    // std::cout << "lo.z+l_cell+iz " << lo.z+l_cell+iz << "\n";

            }
        }
    }
    // Gather field on particle EypBxp from field on grid eypbx_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                EypBxp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    eypbx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz);
            }
        }
    }
    // Gather field on particle Ezp from field on grid ez_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Ezp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    ez_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz);
            }
        }
    }
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Bzp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    bz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz);
            }
        }
    }
    // Gather field on particle Byp from field on grid by_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Byp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    by_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz);
            }
        }
    }
    // Gather field on particle Bxp from field on grid bx_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Bxp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    bx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz);
            }
        }
    }
}



/**
 * \brief Field gather for a single particle
 *
 * \param xp, yp, zp                : Particle position coordinates
 * \param ExmByp, EypBxp, Ezp             : Electric field on particles.
 * \param Bxp, Byp, Bzp             : Magnetic field on particles.
 * \param exmby_arr eypbx_arr ez_arr      : Array4 of the electric field, either full array or tile.
 * \param bx_arr by_arr bz_arr      : Array4 of the magnetic field, either full array or tile.
 * \param ex_type, ey_type, ez_type : IndexType of the electric field
 * \param bx_type, by_type, bz_type : IndexType of the magnetic field
 * \param dx                        : 3D cell spacing
 * \param xyzmin                    : Physical lower bounds of domain in x, y, z.
 * \param lo                        : Index lower bounds of domain.
 * \param nox                       : order of the particle shape function

 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& ExmByp,
                     amrex::ParticleReal& EypBxp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& exmby_arr,
                     amrex::Array4<amrex::Real const> const& eypbx_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::GpuArray<amrex::Real, 3>& dx_arr,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin_arr,
                     const amrex::Dim3& lo,
                     const int depos_order_xy,
                     const int depos_order_z)
{
    if (depos_order_xy == 2 && depos_order_z == 0) {
        doGatherShapeN<2,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx_arr, xyzmin_arr, lo);
    } else if (depos_order_xy == 1 && depos_order_z == 0)  {
        doGatherShapeN<1,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx_arr, xyzmin_arr, lo);
    } else if (depos_order_xy == 3 && depos_order_z == 0)  {
        doGatherShapeN<3,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx_arr, xyzmin_arr, lo);
    }
}

#endif // FIELDGATHER_H_
