/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef FIELDGATHER_H_
#define FIELDGATHER_H_

#include "particles/ShapeFactors.H"

#include <AMReX.H>

/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \tparam depos_order_z Order of the longitudinal shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] zp Particle position z
 * \param[in,out] ExmByp Field on particle: Ex - c*By
 * \param[in,out] EypBxp Field on particle: Ey + c*Bx
 * \param[in,out] Ezp Field on particle: Ez
 * \param[in,out] Bxp Magnetic field on particles: Bx
 * \param[in,out] Byp Magnetic field on particles: By
 * \param[in,out] Bzp Magnetic field on particles: Bz
 * \param[in] exmby_arr field array for field Ex - c*By
 * \param[in] eypbx_arr field array for field  Ey + c*Bx
 * \param[in] ez_arr field array for field  Ez
 * \param[in] bx_arr field array for field  Bx
 * \param[in] by_arr field array for field  By
 * \param[in] bz_arr field array for field  Bz
 * \param[in] dx 3D cell spacing
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
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
                     const amrex::Real x_pos_offset,
                     const amrex::Real y_pos_offset,
                     const amrex::Real z_pos_offset)
{
    // x,y,z direction
    const amrex::Real x = (xp-x_pos_offset)/dx[0];
    const amrex::Real y = (yp-y_pos_offset)/dx[1];
    const amrex::Real z = (zp-z_pos_offset)/dx[2];

    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // z direction;
    amrex::Real sz_cell[depos_order_z + 1];
    const int l_cell = compute_shape_factor<depos_order_z>(sz_cell, z);
    amrex::ignore_unused(l_cell);

    // Gather field on particle ExmByp from field on grid exmby_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                ExmByp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    exmby_arr(j_cell+ix, k_cell+iy, amrex::lbound(exmby_arr).z);
            }
        }
    }
    // Gather field on particle EypBxp from field on grid eypbx_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                EypBxp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    eypbx_arr(j_cell+ix, k_cell+iy, amrex::lbound(eypbx_arr).z);
            }
        }
    }
    // Gather field on particle Ezp from field on grid ez_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Ezp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    ez_arr(j_cell+ix, k_cell+iy, amrex::lbound(ez_arr).z);
            }
        }
    }
    // Gather field on particle Bxp from field on grid bx_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Bxp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    bx_arr(j_cell+ix, k_cell+iy, amrex::lbound(bx_arr).z);
            }
        }
    }
    // Gather field on particle Byp from field on grid by_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Byp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    by_arr(j_cell+ix, k_cell+iy, amrex::lbound(by_arr).z);
            }
        }
    }
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Bzp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    bz_arr(j_cell+ix, k_cell+iy, amrex::lbound(bz_arr).z);
            }
        }
    }
}



/**
 * \brief Field gather for a single particle
 *
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] zp Particle position z
 * \param[in,out] ExmByp Field on particle: Ex - c*By
 * \param[in,out] EypBxp Field on particle: Ey + c*Bx
 * \param[in,out] Ezp Field on particle: Ez
 * \param[in,out] Bxp Magnetic field on particles: Bx
 * \param[in,out] Byp Magnetic field on particles: By
 * \param[in,out] Bzp Magnetic field on particles: Bz
 * \param[in] exmby_arr field array for field Ex - c*By
 * \param[in] eypbx_arr field array for field  Ey + c*Bx
 * \param[in] ez_arr field array for field  Ez
 * \param[in] bx_arr field array for field  Bx
 * \param[in] by_arr field array for field  By
 * \param[in] bz_arr field array for field  Bz
 * \param[in] dx 3D cell spacing
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 * \param[in] depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] depos_order_z Order of the longitudinal shape factor for the field gather
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
                     const amrex::GpuArray<amrex::Real, 3>& dx,
                     const amrex::Real x_pos_offset,
                     const amrex::Real y_pos_offset,
                     const amrex::Real z_pos_offset,
                     const int depos_order_xy,
                     const int depos_order_z)
{
    if (depos_order_xy == 0 && depos_order_z == 0) {
        doGatherShapeN<0,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx, x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 1 && depos_order_z == 0)  {
        doGatherShapeN<1,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx, x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 2 && depos_order_z == 0)  {
        doGatherShapeN<2,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx, x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 3 && depos_order_z == 0)  {
        doGatherShapeN<3,0>(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            dx, x_pos_offset, y_pos_offset, z_pos_offset);
    }
}

/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \tparam depos_order_z Order of the longitudinal shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] zp Particle position z
 * \param[in] Aabssqp |a|^2 field at particle position
 * \param[in] AabssqDxp d/dx |a|^2 field at particle position
 * \param[in] AabssqDyp d/dy |a|^2 field at particle position
 * \param[in] a_arr a field array
 * \param[in] dx 3D cell spacing
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy, int depos_order_z>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::ParticleReal xp,
                          const amrex::ParticleReal yp,
                          const amrex::ParticleReal zp,
                          amrex::ParticleReal& Aabssqp,
                          amrex::ParticleReal& AabssqDxp,
                          amrex::ParticleReal& AabssqDyp,
                          amrex::Array4<amrex::Real const> const& a_arr,
                          const amrex::GpuArray<amrex::Real, 3>& dx,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset,
                          const amrex::Real z_pos_offset)
{
    using namespace amrex::literals;

    // x,y,z direction
    const amrex::Real x = (xp-x_pos_offset)/dx[0];
    const amrex::Real y = (yp-y_pos_offset)/dx[1];
    const amrex::Real z = (zp-z_pos_offset)/dx[2];

    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // z direction;
    amrex::Real sz_cell[depos_order_z + 1];
    const int l_cell = compute_shape_factor<depos_order_z>(sz_cell, z);
    amrex::ignore_unused(l_cell);

    // Gather field Aabssq, AabssqDxp, and AabssqDxp on particle from field on grid a_arr
    // the derivative is calculated on the fly
    // TODO: complex conjugate will be needed when a becomes complex (required for laser evolution)
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Aabssqp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    a_arr(j_cell+ix  , k_cell+iy  , amrex::lbound(a_arr).z)*
                    a_arr(j_cell+ix  , k_cell+iy  , amrex::lbound(a_arr).z);
                AabssqDxp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]/(2.0_rt*dx[0])*
                  ( a_arr(j_cell+ix+1, k_cell+iy  , amrex::lbound(a_arr).z)*
                    a_arr(j_cell+ix+1, k_cell+iy  , amrex::lbound(a_arr).z)
                   -a_arr(j_cell+ix-1, k_cell+iy  , amrex::lbound(a_arr).z)*
                    a_arr(j_cell+ix-1, k_cell+iy  , amrex::lbound(a_arr).z));
                AabssqDyp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]/(2.0_rt*dx[1])*
                  ( a_arr(j_cell+ix  , k_cell+iy+1, amrex::lbound(a_arr).z)*
                    a_arr(j_cell+ix  , k_cell+iy+1, amrex::lbound(a_arr).z)
                   -a_arr(j_cell+ix  , k_cell+iy-1, amrex::lbound(a_arr).z)*
                    a_arr(j_cell+ix  , k_cell+iy-1, amrex::lbound(a_arr).z));
            }
        }
    }
}



/**
 * \brief laser field gather for a single particle
 *
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] zp Particle position z
 * \param[in,out] Aabssqp Field on particle: |a|^2
 * \param[in,out] AabssqDxp Field on particle: d/dx |a|^2
 * \param[in,out] AabssqDyp Field on particle: d/dy |a|^2
 * \param[in] a_arr field array for field a
 * \param[in] dx 3D cell spacing
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 * \param[in] depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] depos_order_z Order of the longitudinal shape factor for the field gather
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& Aabssqp,
                     amrex::ParticleReal& AabssqDxp,
                     amrex::ParticleReal& AabssqDyp,
                     amrex::Array4<amrex::Real const> const& a_arr,
                     const amrex::GpuArray<amrex::Real, 3>& dx,
                     const amrex::Real x_pos_offset,
                     const amrex::Real y_pos_offset,
                     const amrex::Real z_pos_offset,
                     const int depos_order_xy,
                     const int depos_order_z)
{
    if (depos_order_xy == 0 && depos_order_z == 0) {
        doLaserGatherShapeN<0,0>(xp, yp, zp, Aabssqp, AabssqDxp, AabssqDyp, a_arr,
                        dx, x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 1 && depos_order_z == 0)  {
        doLaserGatherShapeN<1,0>(xp, yp, zp, Aabssqp, AabssqDxp, AabssqDyp, a_arr,
                        dx, x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 2 && depos_order_z == 0)  {
        doLaserGatherShapeN<2,0>(xp, yp, zp, Aabssqp, AabssqDxp, AabssqDyp, a_arr,
                        dx, x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 3 && depos_order_z == 0)  {
        doLaserGatherShapeN<3,0>(xp, yp, zp, Aabssqp, AabssqDxp, AabssqDyp, a_arr,
                        dx, x_pos_offset, y_pos_offset, z_pos_offset);
    }
}

/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \tparam depos_order_z Order of the longitudinal shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] zp Particle position z
 * \param[in,out] Aabssqp Field on particle: |a|^2
 * \param[in] a_arr field array for field a
 * \param[in] dx 3D cell spacing
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy, int depos_order_z>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::ParticleReal xp,
                          const amrex::ParticleReal yp,
                          const amrex::ParticleReal zp,
                          amrex::ParticleReal& Aabssqp,
                          amrex::Array4<amrex::Real const> const& a_arr,
                          const amrex::GpuArray<amrex::Real, 3>& dx,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset,
                          const amrex::Real z_pos_offset)
{
    using namespace amrex::literals;

    // x,y,z direction
    const amrex::Real x = (xp-x_pos_offset)/dx[0];
    const amrex::Real y = (yp-y_pos_offset)/dx[1];
    const amrex::Real z = (zp-z_pos_offset)/dx[2];

    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // z direction;
    amrex::Real sz_cell[depos_order_z + 1];
    const int l_cell = compute_shape_factor<depos_order_z>(sz_cell, z);
    amrex::ignore_unused(l_cell);

    // Gather field Aabssq, on particle from field on grid a_arr
    // TODO: complex conjugate will be needed when a becomes complex (required for laser evolution)
    for (int iz=0; iz<=depos_order_z; iz++){
        for (int iy=0; iy<=depos_order_xy; iy++){
            for (int ix=0; ix<=depos_order_xy; ix++){
                Aabssqp += sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*
                    a_arr(j_cell+ix  , k_cell+iy  , amrex::lbound(a_arr).z)*
                    a_arr(j_cell+ix  , k_cell+iy  , amrex::lbound(a_arr).z);
            }
        }
    }
}



/**
 * \brief laser field gather for a single particle
 *
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] zp Particle position z
 * \param[in,out] Aabssqp Field on particle: |a|^2
 * \param[in] a_arr field array for field a
 * \param[in] dx 3D cell spacing
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] z_pos_offset offset for converting positions to indexes
 * \param[in] depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] depos_order_z Order of the longitudinal shape factor for the field gather
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::ParticleReal xp,
                          const amrex::ParticleReal yp,
                          const amrex::ParticleReal zp,
                          amrex::ParticleReal& Aabssqp,
                          amrex::Array4<amrex::Real const> const& a_arr,
                          const amrex::GpuArray<amrex::Real, 3>& dx,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset,
                          const amrex::Real z_pos_offset,
                          const int depos_order_xy,
                          const int depos_order_z)
{
    if (depos_order_xy == 0 && depos_order_z == 0) {
        doLaserGatherShapeN<0,0>(xp, yp, zp, Aabssqp, a_arr, dx,
                                 x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 1 && depos_order_z == 0)  {
        doLaserGatherShapeN<1,0>(xp, yp, zp, Aabssqp, a_arr, dx,
                                 x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 2 && depos_order_z == 0)  {
        doLaserGatherShapeN<2,0>(xp, yp, zp, Aabssqp, a_arr, dx,
                                 x_pos_offset, y_pos_offset, z_pos_offset);
    } else if (depos_order_xy == 3 && depos_order_z == 0)  {
        doLaserGatherShapeN<3,0>(xp, yp, zp, Aabssqp, a_arr, dx,
                                 x_pos_offset, y_pos_offset, z_pos_offset);
    }
}


#endif // FIELDGATHER_H_
