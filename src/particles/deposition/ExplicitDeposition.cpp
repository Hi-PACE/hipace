/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "ExplicitDeposition.H"

#include "Hipace.H"
#include "particles/ShapeFactors.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"

#include "AMReX_GpuLaunch.H"

void
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields, const Laser& laser,
                    amrex::Geometry const& gm, const int lev) {
    HIPACE_PROFILE("ExplicitDeposition()");
    using namespace amrex::literals;

    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti) {

        amrex::FArrayBox& isl_fab = fields.getSlices(lev, WhichSlice::This)[pti];
        const Array3<amrex::Real> arr = isl_fab.array();

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];

        const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
        const int EypBx = Comps[WhichSlice::This]["EypBx"];
        const int Ez = Comps[WhichSlice::This]["Ez"];
        const int Bz = Comps[WhichSlice::This]["Bz"];

        const PhysConst pc = get_phys_const();
        const amrex::Real clight = pc.c;
        // The laser a0 is always normalized
        const amrex::Real a_laser_fac = (pc.m_e/pc.q_e) * (pc.m_e/pc.q_e);

        // Extract particle properties
        const auto& aos = pti.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin();
        const auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        const amrex::Real * const AMREX_RESTRICT wp = soa.GetRealData(PlasmaIdx::w).data();
        const amrex::Real * const AMREX_RESTRICT uxp = soa.GetRealData(PlasmaIdx::ux).data();
        const amrex::Real * const AMREX_RESTRICT uyp = soa.GetRealData(PlasmaIdx::uy).data();
        const amrex::Real * const AMREX_RESTRICT psip = soa.GetRealData(PlasmaIdx::psi).data();

        int const * const AMREX_RESTRICT a_ion_lev =
            plasma.m_can_ionize ? soa.GetIntData(PlasmaIdx::ion_lev).data() : nullptr;

        // Construct empty Array4 with one z slice so that Array3 constructor works for no laser
        const Array3<const amrex::Real> a_laser_arr = laser.m_use_laser ?
            laser.getSlices(WhichLaserSlice::n00j00).const_array(pti) :
            amrex::Array4<const amrex::Real>(nullptr, {0,0,0}, {0,0,1}, 0);

        const amrex::Real * AMREX_RESTRICT dx = gm.CellSize();
        const amrex::Real invvol = Hipace::m_normalized_units ? 1._rt : 1._rt/(dx[0]*dx[1]*dx[2]);
        const amrex::Real dx_inv = 1._rt/dx[0];
        const amrex::Real dy_inv = 1._rt/dx[1];

        const amrex::Real x_pos_offset = GetPosOffset(0, gm, isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, isl_fab.box());

        const amrex::Real charge_invvol_mu0 = plasma.m_charge * invvol * pc.mu0;
        const amrex::Real charge_mass = plasma.m_charge / plasma.m_mass;

        amrex::ParallelFor(
            amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>,      // depos_order
                            amrex::CompileTimeOptions<false, true>,     // can_ionize
                            amrex::CompileTimeOptions<false, true>>{},  // use_laser
            {Hipace::m_depos_order_xy, plasma.m_can_ionize, laser.m_use_laser},
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int ip, auto a_depos_order, auto can_ionize, auto use_laser) {
                constexpr int depos_order = a_depos_order.value;

                const auto position = pos_structs[ip];
                if (position.id() < 0) return;
                const amrex::Real psi = psip[ip];
                const amrex::Real vx = uxp[ip] / (psi * clight);
                const amrex::Real vy = uyp[ip] / (psi * clight);

                amrex::Real q_invvol_mu0 = charge_invvol_mu0;
                amrex::Real q_mass = charge_mass;

                [[maybe_unused]] auto ion_lev = a_ion_lev;
                if constexpr (can_ionize.value) {
                    q_invvol_mu0 *= ion_lev[ip];
                    q_mass *= ion_lev[ip];
                }

                const amrex::Real global_fac = q_invvol_mu0 * wp[ip];

                const amrex::Real xmid = (position.pos(0) - x_pos_offset)*dx_inv;
                amrex::Real sx_cell[depos_order + 1];
                const int i_cell = compute_shape_factor<depos_order>(sx_cell, xmid);

                // y direction
                const amrex::Real ymid = (position.pos(1) - y_pos_offset)*dy_inv;
                amrex::Real sy_cell[depos_order + 1];
                const int j_cell = compute_shape_factor<depos_order>(sy_cell, ymid);

                amrex::Real Aabssqp = 0._rt;
                // Rename variable for NVCC lambda capture to work
                [[maybe_unused]] auto laser_arr = a_laser_arr;
                if constexpr (use_laser.value) {
                    for (int iy=0; iy<=depos_order; iy++){
                        for (int ix=0; ix<=depos_order; ix++){
                            // Its important that Aabssqp is first fully gathered and not used
                            // directly per cell like AabssqDxp and AabssqDyp
                            const amrex::Real x00y00 = abssq(
                                laser_arr(i_cell+ix, j_cell+iy, 0),
                                laser_arr(i_cell+ix, j_cell+iy, 1) );
                            // TODO: fix units
                            Aabssqp += sx_cell[ix]*sy_cell[iy]*x00y00;
                        }
                    }
                }

                // calculate gamma/psi for plasma particles
                const amrex::Real gamma_psi = 0.5_rt * (
                    (1._rt + 0.5_rt * Aabssqp) / (psi * psi)
                    + vx * vx
                    + vy * vy
                    + 1._rt
                );

                for (int iy=0; iy <= depos_order+2; ++iy) {
                    // normal shape factor
                    amrex::Real shape_y = 0._rt;
                    // derivative shape factor
                    amrex::Real shape_dy = 0._rt;
                    if (iy != 0 && iy != depos_order + 2) {
                        shape_y = sy_cell[iy-1] * global_fac;
                    }
                    if (iy < depos_order + 1) {
                        shape_dy = sy_cell[iy];
                    }
                    if (iy > 1) {
                        shape_dy -= sy_cell[iy-2];
                    }
                    shape_dy *= dy_inv * 0.5_rt * clight * global_fac;

                    for (int ix=0; ix <= depos_order+2; ++ix) {
                        amrex::Real shape_x = 0._rt;
                        amrex::Real shape_dx = 0._rt;
                        if (ix != 0 && ix != depos_order + 2) {
                            shape_x = sx_cell[ix-1];
                        }
                        if (ix < depos_order + 1) {
                            shape_dx = sx_cell[ix];
                        }
                        if (ix > 1) {
                            shape_dx -= sx_cell[ix-2];
                        }
                        shape_dx *= dx_inv * 0.5_rt * clight;

                        if ((ix==0 || ix==depos_order + 2) && (iy==0 || iy==depos_order + 2)) {
                            // corners have a shape factor of zero
                            continue;
                        }

                        const int i = i_cell + ix - 1;
                        const int j = j_cell + iy - 1;

                        // get fields per cell instead of gathering them to avoid blurring
                        const amrex::Real Bz_v = arr(i,j,Bz);
                        const amrex::Real Ez_v = arr(i,j,Ez);
                        const amrex::Real ExmBy_v = arr(i,j,ExmBy);
                        const amrex::Real EypBx_v = arr(i,j,EypBx);

                        amrex::Real AabssqDxp = 0._rt;
                        amrex::Real AabssqDyp = 0._rt;
                        [[maybe_unused]] auto laser_fac = a_laser_fac;
                        if constexpr (use_laser.value) {
                            const amrex::Real xp1y00 = abssq(
                                laser_arr(i+1, j  , 0),
                                laser_arr(i+1, j  , 1));
                            const amrex::Real xm1y00 = abssq(
                                laser_arr(i-1, j  , 0),
                                laser_arr(i-1, j  , 1));
                            const amrex::Real x00yp1 = abssq(
                                laser_arr(i  , j+1, 0),
                                laser_arr(i  , j+1, 1));
                            const amrex::Real x00ym1 = abssq(
                                laser_arr(i  , j-1, 0),
                                laser_arr(i  , j-1, 1));
                            AabssqDxp = (xp1y00-xm1y00) * 0.5_rt * dx_inv * laser_fac * clight;
                            AabssqDyp = (x00yp1-x00ym1) * 0.5_rt * dy_inv * laser_fac * clight;
                        }

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, Sy),
                            - shape_x * shape_y * (
                                - Bz_v * vx
                                + ( Ez_v * vy
                                + ExmBy_v * (          - vx * vy)
                                + EypBx_v * (gamma_psi - vy * vy) ) / clight
                                - 0.25_rt * AabssqDyp * q_mass / psi
                            ) * q_mass / psi
                            - shape_dx * shape_y * (
                                - vx * vy
                            )
                            - shape_x * shape_dy * (
                                gamma_psi - vy * vy - 1._rt
                            )
                        );

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, Sx),
                            + shape_x * shape_y * (
                                + Bz_v * vy
                                + ( Ez_v * vx
                                + ExmBy_v * (gamma_psi - vx * vx)
                                + EypBx_v * (          - vx * vy) ) / clight
                                - 0.25_rt * AabssqDxp * q_mass / psi
                            ) * q_mass / psi
                            + shape_dx * shape_y * (
                                gamma_psi - vx * vx - 1._rt
                            )
                            + shape_x * shape_dy * (
                                - vx * vy
                            )
                        );

                    }
                }
            });
    }
}
