/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaParticleAdvance.H"

#include "particles/plasma/PlasmaParticleContainer.H"
#include "particles/particles_utils/FieldGather.H"
#include "PushPlasmaParticles.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"
#include "utils/DualNumbers.H"
#include "particles/particles_utils/ParticleUtil.H"

#include <string>

// explicitly instantiate template to fix wrong warning with gcc
template struct PlasmaMomentumDerivative<amrex::Real>;
template struct PlasmaMomentumDerivative<DualNumber>;

void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, const Fields & fields,
                        amrex::Vector<amrex::Geometry> const& gm, const bool temp_slice,
                        int const lev)
{
    HIPACE_PROFILE("AdvancePlasmaParticles()");
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
    {
        // Extract field array from FabArray
        const amrex::FArrayBox& slice_fab = fields.getSlices(lev)[pti];
        Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
        const int psi_comp = Comps[WhichSlice::This]["Psi"];
        const int ez_comp = Comps[WhichSlice::This]["Ez"];
        const int bx_comp = Comps[WhichSlice::This]["Bx"];
        const int by_comp = Comps[WhichSlice::This]["By"];
        const int bz_comp = Comps[WhichSlice::This]["Bz"];
        const int aabs_comp = Hipace::m_use_laser ? Comps[WhichSlice::This]["aabs"] : -1;

        // Extract properties associated with physical size of the box
        const amrex::Real dx_inv = gm[lev].InvCellSize(0);
        const amrex::Real dy_inv = gm[lev].InvCellSize(1);

        const CheckDomainBounds lev_bounds {gm[lev]};

        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, gm[lev], slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], slice_fab.box());

        // loading the data
        const auto ptd = pti.GetParticleTile().getParticleTileData();

        const bool can_ionize = plasma.m_can_ionize;
        const int n_subcycles = plasma.m_n_subcycles;

        const auto enforceBC = EnforceBC();
        const amrex::Real dz = gm[0].CellSize(2) / n_subcycles;

        if (!temp_slice && lev == 0) {
            // only count particles on non-temp slices and only once for all MR levels
            Hipace::m_num_plasma_particles_pushed += double(pti.numParticles());
        }

        const amrex::Real laser_norm = (plasma.m_charge/phys_const.q_e) * (phys_const.m_e/plasma.m_mass)
            * (plasma.m_charge/phys_const.q_e) * (phys_const.m_e/plasma.m_mass);
        const amrex::Real clight = phys_const.c;
        const amrex::Real clight_inv = 1._rt/phys_const.c;
        const amrex::Real charge_mass_clight_ratio = plasma.m_charge/(plasma.m_mass * phys_const.c);
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        {
            amrex::Long const num_particles = pti.numParticles();
#ifdef AMREX_USE_OMP
            amrex::Long const idx_begin = (num_particles * omp_get_thread_num()) / omp_get_num_threads();
            amrex::Long const idx_end = (num_particles * (omp_get_thread_num()+1)) / omp_get_num_threads();
#else
            amrex::Long constexpr idx_begin = 0;
            amrex::Long const idx_end = num_particles;
#endif

            amrex::ParallelFor(
                amrex::TypeList<
                    amrex::CompileTimeOptions<0, 1, 2, 3>,
                    amrex::CompileTimeOptions<false, true>
                >{}, {
                    Hipace::m_depos_order_xy,
                    Hipace::m_use_laser
                },
                int(idx_end - idx_begin), // int ParallelFor is 3-5% faster than amrex::Long version
                [=] AMREX_GPU_DEVICE (int idx, auto depos_order, auto use_laser) {
                    const int ip = idx + idx_begin;
                    // only push plasma particles on their according MR level
                    if (!ptd.id(ip).is_valid() || ptd.cpu(ip) != lev) return;

                    // define field at particle position reals
                    amrex::Real ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                    amrex::Real Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;
                    amrex::Real Aabssqp = 0._rt, AabssqDxp = 0._rt, AabssqDyp = 0._rt;

                    amrex::Real q_mass_clight_ratio = charge_mass_clight_ratio;
                    amrex::Real laser_norm_ion = laser_norm;
                    if (can_ionize) {
                        q_mass_clight_ratio *= ptd.idata(PlasmaIdx::ion_lev)[ip];
                        laser_norm_ion *=
                            ptd.idata(PlasmaIdx::ion_lev)[ip] * ptd.idata(PlasmaIdx::ion_lev)[ip];
                    }

                    for (int i = 0; i < n_subcycles; i++) {

                        amrex::Real xp = ptd.rdata(PlasmaIdx::x_prev)[ip];
                        amrex::Real yp = ptd.rdata(PlasmaIdx::y_prev)[ip];

                        if (lev == 0 || lev_bounds.contains(xp, yp)) {
                            ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                            Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;
                            Aabssqp = 0._rt, AabssqDxp = 0._rt, AabssqDyp = 0._rt;

                            doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp,
                                    Bzp, slice_arr, psi_comp, ez_comp, bx_comp, by_comp,
                                    bz_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                            if (use_laser.value) {
                                doLaserGatherShapeN<depos_order.value>(xp, yp,
                                    Aabssqp, AabssqDxp, AabssqDyp, slice_arr, aabs_comp,
                                    dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                            }

                            Bxp *= clight;
                            Byp *= clight;
                            Aabssqp *= 0.5_rt * laser_norm_ion;
                            AabssqDxp *= 0.25_rt * clight * laser_norm_ion;
                            AabssqDyp *= 0.25_rt * clight * laser_norm_ion;
                        }

#ifndef HIPACE_USE_AB5_PUSH

                        constexpr int nsub = 4;
                        const amrex::Real sdz = dz/nsub;

                        amrex::Real ux = ptd.rdata(PlasmaIdx::ux_half_step)[ip];
                        amrex::Real uy = ptd.rdata(PlasmaIdx::uy_half_step)[ip];
                        amrex::Real psi = ptd.rdata(PlasmaIdx::psi_half_step)[ip];

                        // full push in momentum
                        // from t-1/2 to t+1/2
                        // using the fields at t
                        for (int isub=0; isub<nsub; ++isub) {

                            const amrex::Real psi_inv = 1._rt/psi;

                            auto [dz_ux, dz_uy, dz_psi] = PlasmaMomentumPush(
                                ux, uy, psi_inv, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                                Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                            const DualNumber ux_dual{ux, dz_ux};
                            const DualNumber uy_dual{uy, dz_uy};
                            const DualNumber psi_inv_dual{psi_inv, -psi_inv*psi_inv*dz_psi};

                            auto [dz_ux_dual, dz_uy_dual, dz_psi_dual] = PlasmaMomentumPush(
                                ux_dual, uy_dual, psi_inv_dual, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                                Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                            ux += sdz*dz_ux + 0.5_rt*sdz*sdz*dz_ux_dual.epsilon;
                            uy += sdz*dz_uy + 0.5_rt*sdz*sdz*dz_uy_dual.epsilon;
                            psi += sdz*dz_psi + 0.5_rt*sdz*sdz*dz_psi_dual.epsilon;

                        }

                        // full push in position
                        // from t to t+1
                        // using the momentum at t+1/2
                        xp += dz*clight_inv*(ux * (1._rt/psi));
                        yp += dz*clight_inv*(uy * (1._rt/psi));

                        if (enforceBC(ptd, ip, xp, yp, ux, uy, PlasmaIdx::w)) return;
                        ptd.pos(0, ip) = xp;
                        ptd.pos(1, ip) = yp;

                        if (!temp_slice) {
                            // update values of the last non temp slice
                            // the next push always starts from these
                            ptd.rdata(PlasmaIdx::ux_half_step)[ip] = ux;
                            ptd.rdata(PlasmaIdx::uy_half_step)[ip] = uy;
                            ptd.rdata(PlasmaIdx::psi_half_step)[ip] = psi;
                            ptd.rdata(PlasmaIdx::x_prev)[ip] = xp;
                            ptd.rdata(PlasmaIdx::y_prev)[ip] = yp;
                        }

                        // half push in momentum
                        // from t+1/2 to t+1
                        // still using the fields at t as an approximation
                        // the result is used for current deposition etc. but not in the pusher
                        for (int isub=0; isub<(nsub/2); ++isub) {

                            const amrex::Real psi_inv = 1._rt/psi;

                            auto [dz_ux, dz_uy, dz_psi] = PlasmaMomentumPush(
                                ux, uy, psi_inv, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                                Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                            const DualNumber ux_dual{ux, dz_ux};
                            const DualNumber uy_dual{uy, dz_uy};
                            const DualNumber psi_inv_dual{psi_inv, -psi_inv*psi_inv*dz_psi};

                            auto [dz_ux_dual, dz_uy_dual, dz_psi_dual] = PlasmaMomentumPush(
                                ux_dual, uy_dual, psi_inv_dual, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                                Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                            ux += sdz*dz_ux + 0.5_rt*sdz*sdz*dz_ux_dual.epsilon;
                            uy += sdz*dz_uy + 0.5_rt*sdz*sdz*dz_uy_dual.epsilon;
                            psi += sdz*dz_psi + 0.5_rt*sdz*sdz*dz_psi_dual.epsilon;

                        }
                        ptd.rdata(PlasmaIdx::ux)[ip] = ux;
                        ptd.rdata(PlasmaIdx::uy)[ip] = uy;
                        ptd.rdata(PlasmaIdx::psi)[ip] = psi;
#else
                        amrex::Real ux = ptd.rdata(PlasmaIdx::ux_half_step)[ip];
                        amrex::Real uy = ptd.rdata(PlasmaIdx::uy_half_step)[ip];
                        amrex::Real psi = ptd.rdata(PlasmaIdx::psi_half_step)[ip];
                        const amrex::Real psi_inv = 1._rt/psi;

                        auto [dz_ux, dz_uy, dz_psi] = PlasmaMomentumPush(
                            ux, uy, psi_inv, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                        ptd.rdata(PlasmaIdx::Fx1)[ip] = clight_inv*(ux * psi_inv);
                        ptd.rdata(PlasmaIdx::Fy1)[ip] = clight_inv*(uy * psi_inv);
                        ptd.rdata(PlasmaIdx::Fux1)[ip] = dz_ux;
                        ptd.rdata(PlasmaIdx::Fuy1)[ip] = dz_uy;
                        ptd.rdata(PlasmaIdx::Fpsi1)[ip] = dz_psi;

                        const amrex::Real ab5_coeffs[5] = {
                            ( 1901._rt / 720._rt ) * dz,    // a1 times dz
                            ( -1387._rt / 360._rt ) * dz,   // a2 times dz
                            ( 109._rt / 30._rt ) * dz,      // a3 times dz
                            ( -637._rt / 360._rt ) * dz,    // a4 times dz
                            ( 251._rt / 720._rt ) * dz      // a5 times dz
                        };

#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                        for (int iab=0; iab<5; ++iab) {
                            xp  += ab5_coeffs[iab] * ptd.rdata(PlasmaIdx::Fx1   + iab)[ip];
                            yp  += ab5_coeffs[iab] * ptd.rdata(PlasmaIdx::Fy1   + iab)[ip];
                            ux  += ab5_coeffs[iab] * ptd.rdata(PlasmaIdx::Fux1  + iab)[ip];
                            uy  += ab5_coeffs[iab] * ptd.rdata(PlasmaIdx::Fuy1  + iab)[ip];
                            psi += ab5_coeffs[iab] * ptd.rdata(PlasmaIdx::Fpsi1 + iab)[ip];
                        }

                        if (enforceBC(ptd, ip, xp, yp, ux, uy, PlasmaIdx::w)) return;
                        ptd.pos(0, ip) = xp;
                        ptd.pos(1, ip) = yp;

                        if (!temp_slice) {
                            // update values of the last non temp slice
                            // the next push always starts from these
                            ptd.rdata(PlasmaIdx::ux_half_step)[ip] = ux;
                            ptd.rdata(PlasmaIdx::uy_half_step)[ip] = uy;
                            ptd.rdata(PlasmaIdx::psi_half_step)[ip] = psi;
                            ptd.rdata(PlasmaIdx::x_prev)[ip] = xp;
                            ptd.rdata(PlasmaIdx::y_prev)[ip] = yp;
                        }

                        ptd.rdata(PlasmaIdx::ux)[ip] = ux;
                        ptd.rdata(PlasmaIdx::uy)[ip] = uy;
                        ptd.rdata(PlasmaIdx::psi)[ip] = psi;
#endif
                    } // loop over subcycles
                });
        }

#ifdef HIPACE_USE_AB5_PUSH
        if (!temp_slice) {
            auto& rd = pti.GetStructOfArrays().GetRealData();

            // shift force terms
            rd[PlasmaIdx::Fx5].swap(rd[PlasmaIdx::Fx4]);
            rd[PlasmaIdx::Fy5].swap(rd[PlasmaIdx::Fy4]);
            rd[PlasmaIdx::Fux5].swap(rd[PlasmaIdx::Fux4]);
            rd[PlasmaIdx::Fuy5].swap(rd[PlasmaIdx::Fuy4]);
            rd[PlasmaIdx::Fpsi5].swap(rd[PlasmaIdx::Fpsi4]);

            rd[PlasmaIdx::Fx4].swap(rd[PlasmaIdx::Fx3]);
            rd[PlasmaIdx::Fy4].swap(rd[PlasmaIdx::Fy3]);
            rd[PlasmaIdx::Fux4].swap(rd[PlasmaIdx::Fux3]);
            rd[PlasmaIdx::Fuy4].swap(rd[PlasmaIdx::Fuy3]);
            rd[PlasmaIdx::Fpsi4].swap(rd[PlasmaIdx::Fpsi3]);

            rd[PlasmaIdx::Fx3].swap(rd[PlasmaIdx::Fx2]);
            rd[PlasmaIdx::Fy3].swap(rd[PlasmaIdx::Fy2]);
            rd[PlasmaIdx::Fux3].swap(rd[PlasmaIdx::Fux2]);
            rd[PlasmaIdx::Fuy3].swap(rd[PlasmaIdx::Fuy2]);
            rd[PlasmaIdx::Fpsi3].swap(rd[PlasmaIdx::Fpsi2]);

            rd[PlasmaIdx::Fx2].swap(rd[PlasmaIdx::Fx1]);
            rd[PlasmaIdx::Fy2].swap(rd[PlasmaIdx::Fy1]);
            rd[PlasmaIdx::Fux2].swap(rd[PlasmaIdx::Fux1]);
            rd[PlasmaIdx::Fuy2].swap(rd[PlasmaIdx::Fuy1]);
            rd[PlasmaIdx::Fpsi2].swap(rd[PlasmaIdx::Fpsi1]);
        }
#endif
    }
}
