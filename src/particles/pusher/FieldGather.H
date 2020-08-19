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
    // const amrex::Real dzi = 1.0/dx[2];

    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    //const amrex::Real zmin = xyzmin[2];

    // constexpr int zdir = (AMREX_SPACEDIM - 1);


    // x,y,z direction
    const amrex::Real x = (xp-xmin)*dxi;
    const amrex::Real y = (yp-ymin)*dyi;
    // z direction
    // const amrex::Real z = (zp-zmin)*dzi;


    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, x - 0.5_rt);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, y - 0.5_rt);

    // // z direction
    // const amrex::Real zmid = (pos_structs[ip].pos(2) - zmin)*dzi;
    // amrex::Real sz_cell[depos_order_z + 1];
    // const int l_cell = compute_shape_factor<depos_order_z>(sz_cell, z - 0.5_rt);


    // amrex::Real sy_node[depos_order + 1];
    // amrex::Real sy_cell[depos_order + 1];
    // amrex::Real sy_node_v[depos_order + 1];
    // amrex::Real sy_cell_v[depos_order + 1];
    // int k_node;
    // int k_cell;
    // int k_node_v;
    // int k_cell_v;
    // if ((ex_type[1] == NODE) || (ez_type[1] == NODE) || (by_type[1] == NODE)) {
    //     k_node = compute_shape_factor(sy_node, y);
    // }
    // if ((ex_type[1] == CELL) || (ez_type[1] == CELL) || (by_type[1] == CELL)) {
    //     k_cell = compute_shape_factor(sy_cell, y - 0.5_rt);
    // }
    // if ((ey_type[1] == NODE) || (bx_type[1] == NODE) || (bz_type[1] == NODE)) {
    //     k_node_v = compute_shape_factor_lower_in_v(sy_node_v, y);
    // }
    // if ((ey_type[1] == CELL) || (bx_type[1] == CELL) || (bz_type[1] == CELL)) {
    //     k_cell_v = compute_shape_factor_lower_in_v(sy_cell_v, y - 0.5_rt);
    // }
    // const amrex::Real (&sy_ex)[depos_order + 1             ] = ((ex_type[1] == NODE) ? sy_node   : sy_cell  );
    // const amrex::Real (&sy_ey)[depos_order + 1             ] = ((ey_type[1] == NODE) ? sy_node_v : sy_cell_v);
    // const amrex::Real (&sy_ez)[depos_order + 1             ] = ((ez_type[1] == NODE) ? sy_node   : sy_cell  );
    // const amrex::Real (&sy_bx)[depos_order + 1             ] = ((bx_type[1] == NODE) ? sy_node_v : sy_cell_v);
    // const amrex::Real (&sy_by)[depos_order + 1             ] = ((by_type[1] == NODE) ? sy_node   : sy_cell  );
    // const amrex::Real (&sy_bz)[depos_order + 1             ] = ((bz_type[1] == NODE) ? sy_node_v : sy_cell_v)
    //
    // // z direction
    // const amrex::Real z = (zp-zmin)*dzi;
    // amrex::Real sz_node[depos_order + 1];
    // amrex::Real sz_cell[depos_order + 1];
    // amrex::Real sz_node_v[depos_order + 1];
    // amrex::Real sz_cell_v[depos_order + 1];
    // int l_node;
    // int l_cell;
    // int l_node_v;
    // int l_cell_v;
    // if ((ex_type[zdir] == NODE) || (ey_type[zdir] == NODE) || (bz_type[zdir] == NODE)) {
    //     l_node = compute_shape_factor(sz_node, z);
    // }
    // if ((ex_type[zdir] == CELL) || (ey_type[zdir] == CELL) || (bz_type[zdir] == CELL)) {
    //     l_cell = compute_shape_factor(sz_cell, z - 0.5_rt);
    // }
    // if ((ez_type[zdir] == NODE) || (bx_type[zdir] == NODE) || (by_type[zdir] == NODE)) {
    //     l_node_v = compute_shape_factor_lower_in_v(sz_node_v, z);
    // }
    // if ((ez_type[zdir] == CELL) || (bx_type[zdir] == CELL) || (by_type[zdir] == CELL)) {
    //     l_cell_v = compute_shape_factor_lower_in_v(sz_cell_v, z - 0.5_rt);
    // }



    // (AMREX_SPACEDIM == 3)
    // Gather field on particle ExmByp from field on grid exmby_arr
    // for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                ExmByp += sx_cell[ix]*sy_cell[iy]*//sz_ex[iz]*
                    exmby_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0); //, lo.z+l_cell+iz);
            }
        }
    // }
    // Gather field on particle EypBxp from field on grid eypbx_arr
    // for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                EypBxp += sx_cell[ix]*sy_cell[iy]*//sz_ey[iz]*
                    eypbx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0); //, lo.z+l_cell+iz);
            }
        }
    // }
    // Gather field on particle Ezp from field on grid ez_arr
    // for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Ezp += sx_cell[ix]*sy_cell[iy]*//sz_ez[iz]*
                    ez_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0); //, lo.z+l_cell+iz);
            }
        }
    // }
    // Gather field on particle Bzp from field on grid bz_arr
    // for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Bzp += sx_cell[ix]*sy_cell[iy]*//sz_bz[iz]*
                    bz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0); //, lo.z+l_cell+iz);
            }
        }
    // }
    // Gather field on particle Byp from field on grid by_arr
    // for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Byp += sx_cell[ix]*sy_cell[iy]*//sz_by[iz]*
                    by_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0); //, lo.z+l_cell+iz);
            }
        }
    // }
    // Gather field on particle Bxp from field on grid bx_arr
    // for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Bxp += sx_cell[ix]*sy_cell[iy]*//sz_bx[iz]*
                    bx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0); //, lo.z+l_cell+iz);
            }
        }
    // }
}

/**
 * \brief Field gather for particles
 *
 * /param getPosition          : A functor for returning the particle position.
 * /param getExternalEField    : A functor for assigning the external E field.
 * /param getExternalBField    : A functor for assigning the external B field.
 * \param ExmByp, EypBxp, Ezp        : Pointer to array of electric field on particles.
 * \param Bxp, Byp, Bzp        : Pointer to array of magnetic field on particles.
 * \param exfab eyfab ezfab    : Array4 of the electric field, either full array or tile.
 * \param ezfab bxfab bzfab    : Array4 of the magnetic field, either full array or tile.
 * \param np_to_gather         : Number of particles for which field is gathered.
 * \param dx                   : 3D cell size
 * \param xyzmin               : Physical lower bounds of domain.
 * \param lo                   : Index lower bounds of domain.
 * \param n_rz_azimuthal_modes : Number of azimuthal modes when using RZ geometry
 */
// template <int depos_order_xy, int depos_order_z>
// void doGatherShapeN(const GetParticlePosition& getPosition,
//                     const GetExternalEField& getExternalE, const GetExternalBField& getExternalB,
//                     amrex::ParticleReal * const ExmByp, amrex::ParticleReal * const EypBxp,
//                     amrex::ParticleReal * const Ezp, amrex::ParticleReal * const Bxp,
//                     amrex::ParticleReal * const Byp, amrex::ParticleReal * const Bzp,
//                     amrex::FArrayBox const * const exfab,
//                     amrex::FArrayBox const * const eyfab,
//                     amrex::FArrayBox const * const ezfab,
//                     amrex::FArrayBox const * const bxfab,
//                     amrex::FArrayBox const * const byfab,
//                     amrex::FArrayBox const * const bzfab,
//                     const long np_to_gather,
//                     const std::array<amrex::Real, 3>& dx,
//                     const std::array<amrex::Real, 3> xyzmin,
//                     const amrex::Dim3 lo,
//                     const long n_rz_azimuthal_modes)
// {
//
//     amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
//     amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};
//
//     amrex::Array4<const amrex::Real> const& exmby_arr = exfab->array();
//     amrex::Array4<const amrex::Real> const& eypbx_arr = eyfab->array();
//     amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
//     amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
//     amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
//     amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();
//
//
//     // Loop over particles and gather fields from
//     // {e,b}{x,y,z}_arr to {E,B}{xyz}p.
//     amrex::ParallelFor(
//         np_to_gather,
//         [=] AMREX_GPU_DEVICE (long ip) {
//
//             amrex::ParticleReal xp, yp, zp;
//             getPosition(ip, xp, yp, zp);
//             getExternalE(ip, ExmByp[ip], EypBxp[ip], Ezp[ip]);
//             getExternalB(ip, Bxp[ip], Byp[ip], Bzp[ip]);
//
//             doGatherShapeN<depos_order_xy, depos_order_z>(
//                 xp, yp, zp, ExmByp[ip], EypBxp[ip], Ezp[ip], Bxp[ip], Byp[ip], Bzp[ip],
//                 exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
//                 ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
//                 dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes);
//         }
//         );
// }

/**
 * \brief Field gather for a single particle
 *
 * /param xp, yp, zp                : Particle position coordinates
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
                     const int nox)
{
    if (nox == 1) {
        doGatherShapeN<1,0>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx_arr, xyzmin_arr, lo);
    } else if (nox == 2) {
        doGatherShapeN<2,0>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx_arr, xyzmin_arr, lo);
    } else if (nox == 3) {
        doGatherShapeN<3,0>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx_arr, xyzmin_arr, lo);
    }
}

#endif // FIELDGATHER_H_
