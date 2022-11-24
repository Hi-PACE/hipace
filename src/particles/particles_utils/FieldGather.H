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

#include "particles/particles_utils/ShapeFactors.H"
#include "utils/GPUUtil.H"
#include <AMReX.H>

/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in,out] ExmByp Field on particle: Ex - c*By
 * \param[in,out] EypBxp Field on particle: Ey + c*Bx
 * \param[in,out] Ezp Field on particle: Ez
 * \param[in,out] Bxp Magnetic field on particles: Bx
 * \param[in,out] Byp Magnetic field on particles: By
 * \param[in,out] Bzp Magnetic field on particles: Bz
 * \param[in] slice_arr slice array for WhichSlice::This
 * \param[in] exmby_comp field component for field Ex - c*By
 * \param[in] eypbx_comp field array for field  Ey + c*Bx
 * \param[in] ez_comp field array for field  Ez
 * \param[in] bx_comp field array for field  Bx
 * \param[in] by_comp field array for field  By
 * \param[in] bz_comp field array for field  Bz
 * \param[in] dx_inv inverse cell spacing in x direction
 * \param[in] dy_inv inverse cell spacing in y direction
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeN (const amrex::Real xp,
                     const amrex::Real yp,
                     amrex::Real& ExmByp,
                     amrex::Real& EypBxp,
                     amrex::Real& Ezp,
                     amrex::Real& Bxp,
                     amrex::Real& Byp,
                     amrex::Real& Bzp,
                     Array3<amrex::Real const> const& slice_arr,
                     const int exmby_comp,
                     const int eypbx_comp,
                     const int ez_comp,
                     const int bx_comp,
                     const int by_comp,
                     const int bz_comp,
                     const amrex::Real dx_inv,
                     const amrex::Real dy_inv,
                     const amrex::Real x_pos_offset,
                     const amrex::Real y_pos_offset)
{
    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int i_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // Gather field on particle from field on grid
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            ExmByp += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, exmby_comp);
            EypBxp += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, eypbx_comp);
            Ezp    += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, ez_comp);
            Bxp    += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, bx_comp);
            Byp    += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, by_comp);
            Bzp    += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, bz_comp);
        }
    }
}



/**
 * \brief Field gather for a single particle
 *
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in,out] ExmByp Field on particle: Ex - c*By
 * \param[in,out] EypBxp Field on particle: Ey + c*Bx
 * \param[in,out] Ezp Field on particle: Ez
 * \param[in,out] Bxp Magnetic field on particles: Bx
 * \param[in,out] Byp Magnetic field on particles: By
 * \param[in,out] Bzp Magnetic field on particles: Bz
 * \param[in] slice_arr slice array for WhichSlice::This
 * \param[in] exmby_comp field component for field Ex - c*By
 * \param[in] eypbx_comp field array for field  Ey + c*Bx
 * \param[in] ez_comp field array for field  Ez
 * \param[in] bx_comp field array for field  Bx
 * \param[in] by_comp field array for field  By
 * \param[in] bz_comp field array for field  Bz
 * \param[in] dx_inv inverse cell spacing in x direction
 * \param[in] dy_inv inverse cell spacing in y direction
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 * \param[in] depos_order_xy Order of the transverse shape factor for the field gather
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
                     Array3<amrex::Real const> const& slice_arr,
                     const int exmby_comp,
                     const int eypbx_comp,
                     const int ez_comp,
                     const int bx_comp,
                     const int by_comp,
                     const int bz_comp,
                     const amrex::Real dx_inv,
                     const amrex::Real dy_inv,
                     const amrex::Real x_pos_offset,
                     const amrex::Real y_pos_offset,
                     const int depos_order_xy)
{
    if (depos_order_xy == 0) {
        doGatherShapeN<0>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, slice_arr,
                          exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp, bz_comp,
                          dx_inv, dy_inv, x_pos_offset, y_pos_offset);
    } else if (depos_order_xy == 1)  {
        doGatherShapeN<1>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, slice_arr,
                          exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp, bz_comp,
                          dx_inv, dy_inv, x_pos_offset, y_pos_offset);
    } else if (depos_order_xy == 2)  {
        doGatherShapeN<2>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, slice_arr,
                          exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp, bz_comp,
                          dx_inv, dy_inv, x_pos_offset, y_pos_offset);
    } else if (depos_order_xy == 3)  {
        doGatherShapeN<3>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, slice_arr,
                          exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp, bz_comp,
                          dx_inv, dy_inv, x_pos_offset, y_pos_offset);
    }
}



/**
 * \brief Field gather for a single particle of just Bx and By
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in,out] Bxp Magnetic field on particles: Bx
 * \param[in,out] Byp Magnetic field on particles: By
 * \param[in] slice_arr slice array for WhichSlice::This
 * \param[in] bx_comp field array for field  Bx
 * \param[in] by_comp field array for field  By
 * \param[in] dx_inv inverse cell spacing in x direction
 * \param[in] dy_inv inverse cell spacing in y direction
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doBxByGatherShapeN (const amrex::Real xp,
                         const amrex::Real yp,
                         amrex::Real& Bxp,
                         amrex::Real& Byp,
                         Array3<amrex::Real const> const& slice_arr,
                         const int bx_comp,
                         const int by_comp,
                         const amrex::Real dx_inv,
                         const amrex::Real dy_inv,
                         const amrex::Real x_pos_offset,
                         const amrex::Real y_pos_offset)
{
    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int i_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // Gather field on particle from field on grid
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            Bxp += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, bx_comp);
            Byp += sx_cell[ix]*sy_cell[iy] * slice_arr(i_cell+ix, j_cell+iy, by_comp);
        }
    }
}



/**
 * \brief Laser field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] Aabssqp |a|^2 field at particle position
 * \param[in] AabssqDxp d/dx |a|^2 field at particle position
 * \param[in] AabssqDyp d/dy |a|^2 field at particle position
 * \param[in] a_arr a field array
 * \param[in] dx_inv inverse cell spacing in x direction
 * \param[in] dy_inv inverse cell spacing in y direction
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::Real xp,
                          const amrex::Real yp,
                          amrex::Real& Aabssqp,
                          amrex::Real& AabssqDxp,
                          amrex::Real& AabssqDyp,
                          Array3<amrex::Real const> const& a_arr,
                          const amrex::Real dx_inv,
                          const amrex::Real dy_inv,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset)
{
    using namespace amrex::literals;

    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // --- Compute shape factors
    // x direction
    // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int i_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // Gather field Aabssq, AabssqDxp, and AabssqDxp on particle from field on grid a_arr
    // the derivative is calculated on the fly
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            const amrex::Real x00y00 = abssq(
                a_arr(i_cell+ix  , j_cell+iy  , 0),
                a_arr(i_cell+ix  , j_cell+iy  , 1) );
            const amrex::Real xp1y00 = abssq(
                a_arr(i_cell+ix+1, j_cell+iy  , 0),
                a_arr(i_cell+ix+1, j_cell+iy  , 1) );
            const amrex::Real xm1y00 = abssq(
                a_arr(i_cell+ix-1, j_cell+iy  , 0),
                a_arr(i_cell+ix-1, j_cell+iy  , 1) );
            const amrex::Real x00yp1 = abssq(
                a_arr(i_cell+ix  , j_cell+iy+1, 0),
                a_arr(i_cell+ix  , j_cell+iy+1, 1) );
            const amrex::Real x00ym1 = abssq(
                a_arr(i_cell+ix  , j_cell+iy-1, 0),
                a_arr(i_cell+ix  , j_cell+iy-1, 1) );
            Aabssqp += sx_cell[ix]*sy_cell[iy]*x00y00;
            AabssqDxp += sx_cell[ix]*sy_cell[iy]*0.5_rt*dx_inv*(xp1y00-xm1y00);
            AabssqDyp += sx_cell[ix]*sy_cell[iy]*0.5_rt*dy_inv*(x00yp1-x00ym1);
        }
    }
}



/**
 * \brief Laser field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in,out] Aabssqp Field on particle: |a|^2
 * \param[in] a_arr field array for field a
 * \param[in] dx_inv inverse cell spacing in x direction
 * \param[in] dy_inv inverse cell spacing in y direction
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::Real xp,
                          const amrex::Real yp,
                          amrex::Real& Aabssqp,
                          Array3<amrex::Real const> const& a_arr,
                          const amrex::Real dx_inv,
                          const amrex::Real dy_inv,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset)
{
    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // --- Compute shape factors
    // x direction
    // i_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
    amrex::Real sx_cell[depos_order_xy + 1];
    const int i_cell = compute_shape_factor<depos_order_xy>(sx_cell, x);

    // y direction
    amrex::Real sy_cell[depos_order_xy + 1];
    const int j_cell = compute_shape_factor<depos_order_xy>(sy_cell, y);

    // Gather field Aabssq, on particle from field on grid a_arr
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            const amrex::Real x00y00 = abssq(
                a_arr(i_cell+ix  , j_cell+iy  , 0),
                a_arr(i_cell+ix  , j_cell+iy  , 1) );
            Aabssqp += sx_cell[ix]*sy_cell[iy]*x00y00;
        }
    }
}

#endif // FIELDGATHER_H_
