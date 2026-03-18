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
 * \param[in] psi_comp field component for field Psi
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
                     const int psi_comp,
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

    // Gather field on particle from field on grid

    HIPACE_LOOP_UNROLL
    for (int iy=0; iy<=depos_order_xy; iy++){
        HIPACE_LOOP_UNROLL
        for (int ix=0; ix<=depos_order_xy; ix++){

            // Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order_xy>(y, iy);
            auto [shape_x, i] = shape_factor<depos_order_xy>(x, ix);

            Ezp += (shape_x * shape_y) * slice_arr(i, j, ez_comp);
            Bxp += (shape_x * shape_y) * slice_arr(i, j, bx_comp);
            Byp += (shape_x * shape_y) * slice_arr(i, j, by_comp);
            Bzp += (shape_x * shape_y) * slice_arr(i, j, bz_comp);
        }
    }

    constexpr int derivative_type = 1;

    HIPACE_LOOP_UNROLL
    for (int iy=0; iy<=depos_order_xy+derivative_type; iy++){
        HIPACE_LOOP_UNROLL
        for (int ix=0; ix<=depos_order_xy+derivative_type; ix++){

            // Compute shape factors
            auto [shape_y, shape_dy, j] =
                derivative_shape_factor<derivative_type, depos_order_xy>(y, iy);
            auto [shape_x, shape_dx, i] =
                derivative_shape_factor<derivative_type, depos_order_xy>(x, ix);

            const amrex::Real psip = slice_arr(i, j, psi_comp);
            ExmByp += (shape_dx * shape_y) * psip * dx_inv;
            EypBxp += (shape_x * shape_dy) * psip * dy_inv;
        }
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

    // Gather field on particle from field on grid
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            // --- Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order_xy>(y, iy);
            auto [shape_x, i] = shape_factor<depos_order_xy>(x, ix);

            Bxp += shape_x * shape_y * slice_arr(i, j, bx_comp);
            Byp += shape_x * shape_y * slice_arr(i, j, by_comp);
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
 * \param[in] slice_arr slice array for WhichSlice::This
 * \param[in] aabs_comp field component for field aabs
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
                          Array3<amrex::Real const> const& slice_arr,
                          const int aabs_comp,
                          const amrex::Real dx_inv,
                          const amrex::Real dy_inv,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset)
{
    using namespace amrex::literals;

    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // Gather field Aabssq, AabssqDxp, and AabssqDxp on particle from field on grid slice_arr
    // the derivative is calculated on the fly
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            // --- Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order_xy>(y, iy);
            auto [shape_x, i] = shape_factor<depos_order_xy>(x, ix);

            const amrex::Real x00y00 = slice_arr(i  , j  , aabs_comp);
            const amrex::Real xp1y00 = slice_arr(i+1, j  , aabs_comp);
            const amrex::Real xm1y00 = slice_arr(i-1, j  , aabs_comp);
            const amrex::Real x00yp1 = slice_arr(i  , j+1, aabs_comp);
            const amrex::Real x00ym1 = slice_arr(i  , j-1, aabs_comp);
            Aabssqp += shape_x * shape_y * x00y00;
            AabssqDxp += shape_x * shape_y * 0.5_rt * dx_inv * (xp1y00 - xm1y00);
            AabssqDyp += shape_x * shape_y * 0.5_rt * dy_inv * (x00yp1 - x00ym1);
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
 * \param[in] slice_arr slice array for WhichSlice::This
 * \param[in] aabs_comp field component for field aabs
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
                          Array3<const amrex::Real> const& slice_arr,
                          const int aabs_comp,
                          const amrex::Real dx_inv,
                          const amrex::Real dy_inv,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset)
{
    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // Gather field Aabssq, on particle from field on grid slice_arr
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            // --- Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order_xy>(y, iy);
            auto [shape_x, i] = shape_factor<depos_order_xy>(x, ix);

            const amrex::Real x00y00 = slice_arr(i, j, aabs_comp);
            Aabssqp += shape_x * shape_y * x00y00;
        }
    }
}

/**
 * \brief Laser field gather for a single particle
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the field gather
 * \param[in] xp Particle position x
 * \param[in] yp Particle position y
 * \param[in] Ap a field at particle position
 * \param[in] ADxp d/dx a field at particle position
 * \param[in] ADzetap d/dzeta a field at particle position
 * \param[in] laser_arr slice array for WhichSlice::This
 * \param[in] dx_inv inverse cell spacing in x direction
 * \param[in] dy_inv inverse cell spacing in y direction
 * \param[in] dzeta_inv inverse cell spacing in zeta direction
 * \param[in] x_pos_offset offset for converting positions to indexes
 * \param[in] y_pos_offset offset for converting positions to indexes
 */
template <int depos_order_xy>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doLaserGatherShapeN (const amrex::Real xp,
                          const amrex::Real yp,
                          amrex::GpuComplex<amrex::Real>& Ap,
                          amrex::GpuComplex<amrex::Real>& ADxp,
                          amrex::GpuComplex<amrex::Real>& ADzetap,
                          Array3<amrex::Real const> const& laser_arr,
                          const amrex::Real dx_inv,
                          const amrex::Real dy_inv,
                          const amrex::Real dzeta_inv,
                          const amrex::Real x_pos_offset,
                          const amrex::Real y_pos_offset)
{
    using namespace amrex::literals;

    // x,y direction
    const amrex::Real x = (xp-x_pos_offset)*dx_inv;
    const amrex::Real y = (yp-y_pos_offset)*dy_inv;

    // Gather field Aabssq, AabssqDxp, and AabssqDxp on particle from field on grid slice_arr
    // the derivative is calculated on the fly
    for (int iy=0; iy<=depos_order_xy; iy++){
        for (int ix=0; ix<=depos_order_xy; ix++){
            using namespace WhichLaserSlice;

            // --- Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order_xy>(y, iy);
            auto [shape_x, i] = shape_factor<depos_order_xy>(x, ix);

            const amrex::Real s_cell = shape_x * shape_y;
            Ap += amrex::GpuComplex<amrex::Real> {
                s_cell * laser_arr(i, j, n00j00_r),
                s_cell * laser_arr(i, j, n00j00_i)
            };
            ADxp += amrex::GpuComplex<amrex::Real> {
                s_cell * 0.5_rt * dx_inv *
                    (laser_arr(i + 1, j, n00j00_r) - laser_arr(i - 1, j, n00j00_r)),
                s_cell * 0.5_rt * dx_inv *
                    (laser_arr(i + 1, j, n00j00_i) - laser_arr(i - 1, j, n00j00_i))
            };
            ADzetap += amrex::GpuComplex<amrex::Real> {
                s_cell * dzeta_inv *
                    (laser_arr(i, j, n00jp1_r) - laser_arr(i, j, n00j00_r)),
                s_cell * dzeta_inv *
                    (laser_arr(i, j, n00jp1_i) - laser_arr(i, j, n00j00_i))
            };
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherEz (const amrex::Real xp,
                     const amrex::Real yp,
                     amrex::Real& Ezp,
                     Array3<amrex::Real const> const& slice_arr,
                     const int ez_comp,
                     const amrex::Real dx_inv,
                     const amrex::Real dy_inv,
                     const amrex::Real x_pos_offset,
                     const amrex::Real y_pos_offset)
{
    // x,y direction
    const amrex::Real x = (xp - x_pos_offset) * dx_inv;
    const amrex::Real y = (yp - y_pos_offset) * dy_inv;

    // Compute shape factors
    auto [shape_y, j] = shape_factor<0>(y, 0);
    auto [shape_x, i] = shape_factor<0>(x, 0);

    // Gather Ez on particle from field on grid
    Ezp += (shape_x * shape_y) * slice_arr(i, j, ez_comp);
}

#endif // FIELDGATHER_H_
