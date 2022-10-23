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
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields,
                    amrex::Geometry const& gm, const int lev) {
    HIPACE_PROFILE("ExplicitDeposition()");
    using namespace amrex::literals;

    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti) {

        amrex::FArrayBox& isl_fab = fields.getSlices(lev, WhichSlice::This)[pti];
        Array3<amrex::Real> arr = isl_fab.array();

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];

        const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
        const int EypBx = Comps[WhichSlice::This]["EypBx"];
        const int Ez = Comps[WhichSlice::This]["Ez"];
        const int Bz = Comps[WhichSlice::This]["Bz"];

        PhysConst pc = get_phys_const();

        // Extract particle properties
        auto& aos = pti.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin();
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        const amrex::Real * const AMREX_RESTRICT wp = soa.GetRealData(PlasmaIdx::w).data();
        const amrex::Real * const AMREX_RESTRICT uxp = soa.GetRealData(PlasmaIdx::ux).data();
        const amrex::Real * const AMREX_RESTRICT uyp = soa.GetRealData(PlasmaIdx::uy).data();
        const amrex::Real * const AMREX_RESTRICT psip = soa.GetRealData(PlasmaIdx::psi).data();

        amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
        const amrex::Real invvol = Hipace::m_normalized_units ? 1._rt : 1._rt/(dx[0]*dx[1]*dx[2]);
        const amrex::Real dx_inv = 1._rt/dx[0];
        const amrex::Real dy_inv = 1._rt/dx[1];

        const amrex::Real x_pos_offset = GetPosOffset(0, gm, isl_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, isl_fab.box());

        const amrex::Real charge = plasma.m_charge;
        const amrex::Real mass = plasma.m_mass;

        amrex::ParallelFor(
            amrex::TypeList<amrex::CompileTimeOptions<0,1,2,3>>{},
            {Hipace::m_depos_order_xy},
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int i, auto a_depos_order) {
                constexpr int depos_order = a_depos_order.value;

                const auto position = pos_structs[i];
                if (position.id() < 0) return;
                const amrex::Reak w = wp[i];
                const amrex::Real ux = uxp[i];
                const amrex::Real uy = uyp[i];
                const amrex::Real psi = psip[i];
                const amrex::Real psi_c = psi * pc.c;

                const amrex::Real xmid = (position.pos(0) - x_pos_offset)*dx_inv;
                amrex::Real sx_cell[depos_order + 1];
                const int i_cell = compute_shape_factor<depos_order>(sx_cell, xmid) - 1;

                // y direction
                const amrex::Real ymid = (position.pos(1) - y_pos_offset)*dy_inv;
                amrex::Real sy_cell[depos_order + 1];
                const int j_cell = compute_shape_factor<depos_order>(sy_cell, ymid) - 1;

                amrex::Real global_fac = w * charge * invvol * pc.mu0 / psi;

                for (int iy=0; iy <= depos_order+2; ++iy) {
                    amrex::Real shape_y = 0._rt;
                    amrex::Real shape_dx = 0._rt;
                    if (ix != 0 && ix != depos_order + 2) {
                        shape_y = j_cell[iy-1] * global_fac;
                    }
                    if (ix < depos_order + 1) {
                        shape_dy = j_cell[iy];
                    }
                    if (ix > 1) {
                        shape_dy -= j_cell[iy-2];
                    }
                    shape_dy *= dy_inv * 0.5_rt * global_fac;

                    for (int ix=0; ix <= depos_order+2; ++ix) {
                        amrex::Real shape_x = 0._rt;
                        amrex::Real shape_dx = 0._rt;
                        if (ix != 0 && ix != depos_order + 2) {
                            shape_x = i_cell[ix-1];
                        }
                        if (ix < depos_order + 1) {
                            shape_dx = i_cell[ix];
                        }
                        if (ix > 1) {
                            shape_dx -= i_cell[ix-2];
                        }
                        shape_dx *= dx_inv * 0.5_rt;

                        const int i = i_cell + ix;
                        const int j = j_cell + iy;

                        // calculate gamma for plasma particles
                        const amrex::Real gamma_c = (pc.c * pc.c + ux * ux + uy * uy
                                                    + psi_c * psi_c) / (2._rt * psi_c);

                        const amrex::Real Bz_cv = arr(i,j,Bz) * pc.c;
                        const amrex::Real Ez_v = arr(i,j,Ez);
                        const amrex::Real ExmBy_v = arr(i,j,ExmBy);
                        const amrex::Real EypBx_v = arr(i,j,EypBx);

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, Sx),
                            shape_x * shape_y * (
                                - Bz_cv * uy
                                - Ez_v * ux
                                + ExmBy_v * (gamma_c - ux * ux / psi_c)
                                + EypBx_v * (        - ux * uy / psi_c)
                            ) * charge / (psi_c * mass * pc.c)
                            + shape_dx * shape_y * (
                                (gamma_c - ux * ux / psi_c) - psi_c
                            )
                            + shape_x * shape_dy * (
                                - ux * uy / psi_c
                            )
                        );

                        amrex::Gpu::Atomic::Add(arr.ptr(i, j, Sy),
                            shape_x * shape_y * (
                                + Bz_cv * ux
                                - Ez_v * uy
                                + ExmBy_v * (        - ux * uy / psi_c)
                                + EypBx_v * (gamma_c - uy * uy / psi_c)
                            ) * charge / (psi_c * mass * pc.c)
                            + shape_dx * shape_y * (
                                - uy * ux / psi_c
                            )
                            + shape_x * shape_dy * (
                                (gamma_c - uy * uy / psi_c) - psi_c
                            )
                        );

                    }
                }
            });

    }
}
