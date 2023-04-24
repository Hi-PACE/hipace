/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "ExplicitDeposition.H"

#include "Hipace.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"

#include "AMReX_GpuLaunch.H"

void
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields, const MultiLaser& multi_laser,
                    amrex::Vector<amrex::Geometry> const& gm, const int lev) {
    HIPACE_PROFILE("ExplicitDeposition()");
    using namespace amrex::literals;

    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {

        amrex::FArrayBox& isl_fab = fields.getSlices(lev)[pti];
        const Array3<amrex::Real> arr = isl_fab.array();

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];

        const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
        const int EypBx = Comps[WhichSlice::This]["EypBx"];
        const int Ez = Comps[WhichSlice::This]["Ez"];
        const int Bz = Comps[WhichSlice::This]["Bz"];

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

        // Construct empty Array4 if there is no laser
        const Array3<const amrex::Real> a_laser_arr = multi_laser.m_use_laser ?
            multi_laser.getSlices().const_array(pti, WhichLaserSlice::n00j00_r) :
            amrex::Array4<const amrex::Real>();

        const amrex::Real x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

        const amrex::Real * AMREX_RESTRICT dx = gm[lev].CellSize();
        const amrex::Real invvol = Hipace::m_normalized_units ?
            gm[0].CellSize(0)*gm[0].CellSize(1) / (gm[lev].CellSize(0)*gm[lev].CellSize(1))
            : 1._rt/(dx[0]*dx[1]*dx[2]);
        const amrex::Real dx_inv = 1._rt/dx[0];
        const amrex::Real dy_inv = 1._rt/dx[1];

        const PhysConst pc = get_phys_const();
        const amrex::Real a_clight = pc.c;
        const amrex::Real clight_inv = 1._rt/pc.c;
        // The laser a0 is always normalized
        const amrex::Real a_laser_fac = (pc.m_e/pc.q_e) * (pc.m_e/pc.q_e);
        const amrex::Real charge_invvol_mu0 = plasma.m_charge * invvol * pc.mu0;
        const amrex::Real charge_mass_ratio = plasma.m_charge / plasma.m_mass;

        amrex::ParallelFor(
            amrex::TypeList<
                amrex::CompileTimeOptions<0, 1, 2, 3>,  // depos_order
                amrex::CompileTimeOptions<0, 1, 2>,     // derivative_type
                amrex::CompileTimeOptions<false, true>, // can_ionize
                amrex::CompileTimeOptions<false, true>  // use_laser
            >{}, {
                Hipace::m_depos_order_xy,
                Hipace::m_depos_derivative_type,
                plasma.m_can_ionize,
                multi_laser.m_use_laser
            },
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int ip, auto a_depos_order, auto a_derivative_type,
                                          auto can_ionize, auto use_laser) noexcept {
                constexpr int depos_order = a_depos_order.value;
                constexpr int derivative_type = a_derivative_type.value;

                const auto positions = pos_structs[ip];
                if (positions.id() < 0 || positions.cpu() < lev) return;
                const amrex::Real psi_inv = 1._rt/psip[ip];
                const amrex::Real xp = positions.pos(0);
                const amrex::Real yp = positions.pos(1);
                const amrex::Real vx = uxp[ip] * psi_inv * clight_inv;
                const amrex::Real vy = uyp[ip] * psi_inv * clight_inv;

                amrex::Real q_invvol_mu0 = charge_invvol_mu0;
                amrex::Real q_mass_ratio = charge_mass_ratio;

                // Rename variable for NVCC lambda capture to work
                [[maybe_unused]] auto ion_lev = a_ion_lev;
                if constexpr (can_ionize.value) {
                    q_invvol_mu0 *= ion_lev[ip];
                    q_mass_ratio *= ion_lev[ip];
                }

                const amrex::Real charge_density_mu0 = q_invvol_mu0 * wp[ip];

                const amrex::Real xmid = (xp - x_pos_offset) * dx_inv;
                const amrex::Real ymid = (yp - y_pos_offset) * dy_inv;

                amrex::Real Aabssqp = 0._rt;
                // Rename variable for NVCC lambda capture to work
                [[maybe_unused]] auto laser_arr = a_laser_arr;
                if constexpr (use_laser.value) {
                    // Its important that Aabssqp is first fully gathered and not used
                    // directly per cell like AabssqDxp and AabssqDyp
                    doLaserGatherShapeN<depos_order>(xp, yp, Aabssqp, laser_arr,
                                                     dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                }

                // calculate gamma/psi for plasma particles
                const amrex::Real gamma_psi = 0.5_rt * (
                    (1._rt + 0.5_rt * Aabssqp) * psi_inv * psi_inv // TODO: fix units
                    + vx * vx
                    + vy * vy
                    + 1._rt
                );

#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int iy=0; iy <= depos_order+derivative_type; ++iy) {
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                    for (int ix=0; ix <= depos_order+derivative_type; ++ix) {

                        if constexpr (derivative_type == 2) {
                            if ((ix==0 || ix==depos_order + 2) && (iy==0 || iy==depos_order + 2)) {
                                // corners have a shape factor of zero
                                continue;
                            }
                        }

                        auto [shape_y, shape_dy, j] =
                            single_derivative_shape_factor<derivative_type, depos_order>(ymid, iy);
                        auto [shape_x, shape_dx, i] =
                            single_derivative_shape_factor<derivative_type, depos_order>(xmid, ix);

                        // get fields per cell instead of gathering them to avoid blurring
                        const amrex::Real Bz_v = arr(i,j,Bz);
                        const amrex::Real Ez_v = arr(i,j,Ez);
                        const amrex::Real ExmBy_v = arr(i,j,ExmBy);
                        const amrex::Real EypBx_v = arr(i,j,EypBx);

                        amrex::Real AabssqDxp = 0._rt;
                        amrex::Real AabssqDyp = 0._rt;
                        // Rename variables for NVCC lambda capture to work
                        [[maybe_unused]] auto laser_fac = a_laser_fac;
                        [[maybe_unused]] auto clight = a_clight;
                        if constexpr (use_laser.value) {
                            // avoid going outside of domain
                            if (shape_x * shape_y != 0._rt) {
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
                        }

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, Sy), charge_density_mu0 * (
                            - shape_x * shape_y * (
                                - Bz_v * vx
                                + ( Ez_v * vy
                                + ExmBy_v * (          - vx * vy)
                                + EypBx_v * (gamma_psi - vy * vy) ) * clight_inv
                                - 0.25_rt * AabssqDyp * q_mass_ratio * psi_inv
                            ) * q_mass_ratio * psi_inv
                            + ( - shape_dx * shape_y * dx_inv * (
                                - vx * vy
                            )
                            - shape_x * shape_dy * dy_inv * (
                                gamma_psi - vy * vy - 1._rt
                            )) * a_clight
                        ));

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, Sx), charge_density_mu0 * (
                            + shape_x * shape_y * (
                                + Bz_v * vy
                                + ( Ez_v * vx
                                + ExmBy_v * (gamma_psi - vx * vx)
                                + EypBx_v * (          - vx * vy) ) * clight_inv
                                - 0.25_rt * AabssqDxp * q_mass_ratio * psi_inv
                            ) * q_mass_ratio * psi_inv
                            + ( + shape_dx * shape_y * dx_inv * (
                                gamma_psi - vx * vx - 1._rt
                            )
                            + shape_x * shape_dy * dy_inv * (
                                - vx * vy
                            )) * a_clight
                        ));
                    }
                }
            });
    }
}
