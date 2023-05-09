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
                        amrex::Vector<amrex::Geometry> const& gm, const bool temp_slice, int const lev,
                        PlasmaBins& bins, const MultiLaser& multi_laser)
{
    HIPACE_PROFILE("AdvancePlasmaParticles()");
    using namespace amrex::literals;

    const bool do_tiling = Hipace::m_do_tiling;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm[lev].CellSize();
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

        // extract the laser Fields
        const bool use_laser = multi_laser.m_use_laser;
        const amrex::MultiFab& a_mf = multi_laser.getSlices();

        // Extract field array from MultiFab
        Array3<const amrex::Real> const& a_arr = use_laser ?
            a_mf[pti].const_array(WhichLaserSlice::n00j00_r) : amrex::Array4<const amrex::Real>();

        const amrex::Real dx_inv = 1._rt/dx[0];
        const amrex::Real dy_inv = 1._rt/dx[1];

        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, gm[lev], slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm[lev], slice_fab.box());

        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        // loading the data
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_half_step = soa.GetRealData(PlasmaIdx::ux_half_step).data();
        amrex::Real * const uy_half_step = soa.GetRealData(PlasmaIdx::uy_half_step).data();
        amrex::Real * const psi_half_step =soa.GetRealData(PlasmaIdx::psi_half_step).data();
#ifdef HIPACE_USE_AB5_PUSH
        auto arrdata = soa.realarray();
#endif
        int * const ion_lev = plasma.m_can_ionize ? soa.GetIntData(PlasmaIdx::ion_lev).data()
                                                  : nullptr;

        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto setPositionEnforceBC = EnforceBCandSetPos<PTileType>(pti.GetParticleTile(), gm[0]);
        const amrex::Real dz = gm[0].CellSize(2);

        const amrex::Real me_clight_mass_ratio = phys_const.c * phys_const.m_e/plasma.m_mass;
        const amrex::Real clight = phys_const.c;
        const amrex::Real clight_inv = 1._rt/phys_const.c;
        const amrex::Real charge_mass_clight_ratio = plasma.m_charge/(plasma.m_mass * phys_const.c);

        const int ntiles = do_tiling ? bins.numBins() : 1;

#ifdef AMREX_USE_OMP
#pragma omp parallel for if (amrex::Gpu::notInLaunchRegion())
#endif
        for (int itile=0; itile<ntiles; itile++){
            BeamBins::index_type const * const indices =
                do_tiling ? bins.permutationPtr() : nullptr;
            BeamBins::index_type const * const offsets =
                do_tiling ? bins.offsetsPtr() : nullptr;
            int const num_particles =
                do_tiling ? offsets[itile+1]-offsets[itile] : pti.numParticles();
            amrex::ParallelFor(
                amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
                {Hipace::m_depos_order_xy},
                num_particles,
                [=] AMREX_GPU_DEVICE (long idx, auto depos_order) {
                    const int ip = do_tiling ? indices[offsets[itile]+idx] : idx;
                    // only push plasma particles on their according MR level
                    if (setPositionEnforceBC.m_pardata.id(ip) < 0
                        setPositionEnforceBC.m_pardata.cpu(ip) != lev) return;

                    amrex::Real xp = x_prev[ip];
                    amrex::Real yp = y_prev[ip];

                    // define field at particle position reals
                    amrex::Real ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                    amrex::Real Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;
                    amrex::Real Aabssqp = 0._rt, AabssqDxp = 0._rt, AabssqDyp = 0._rt;

                    doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp,
                            Bzp, slice_arr, psi_comp, ez_comp, bx_comp, by_comp,
                            bz_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                    if (use_laser) {
                        doLaserGatherShapeN<depos_order.value>(xp, yp, Aabssqp, AabssqDxp,
                            AabssqDyp, a_arr, dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                    }

                    amrex::Real q_mass_clight_ratio = charge_mass_clight_ratio;
                    if (ion_lev) {
                        q_mass_clight_ratio *= ion_lev[ip];
                    }
                    Bxp *= clight;
                    Byp *= clight;
                    Aabssqp *= 0.5_rt; // TODO: fix units of Aabssqp
                    AabssqDxp *= 0.25_rt * me_clight_mass_ratio;
                    AabssqDyp *= 0.25_rt * me_clight_mass_ratio;

#ifndef HIPACE_USE_AB5_PUSH

                    constexpr int nsub = 4;
                    const amrex::Real sdz = dz/nsub;

                    amrex::Real ux = ux_half_step[ip];
                    amrex::Real uy = uy_half_step[ip];
                    amrex::Real psi = psi_half_step[ip];

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

                    if (setPositionEnforceBC(ip, xp, yp)) return;

                    if (!temp_slice) {
                        // update values of the last non temp slice
                        // the next push always starts from these
                        ux_half_step[ip] = ux;
                        uy_half_step[ip] = uy;
                        psi_half_step[ip] = psi;
                        x_prev[ip] = xp;
                        y_prev[ip] = yp;
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
                    uxp[ip] = ux;
                    uyp[ip] = uy;
                    psip[ip] = psi;
#else
                    amrex::Real ux = ux_half_step[ip];
                    amrex::Real uy = uy_half_step[ip];
                    amrex::Real psi = psi_half_step[ip];
                    const amrex::Real psi_inv = 1._rt/psi;

                    auto [dz_ux, dz_uy, dz_psi] = PlasmaMomentumPush(
                        ux, uy, psi_inv, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                        Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                    arrdata[PlasmaIdx::Fx1][ip] = clight_inv*(ux * psi_inv);
                    arrdata[PlasmaIdx::Fy1][ip] = clight_inv*(uy * psi_inv);
                    arrdata[PlasmaIdx::Fux1][ip] = dz_ux;
                    arrdata[PlasmaIdx::Fuy1][ip] = dz_uy;
                    arrdata[PlasmaIdx::Fpsi1][ip] = dz_psi;

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
                        xp  += ab5_coeffs[iab] * arrdata[PlasmaIdx::Fx1   + iab][ip];
                        yp  += ab5_coeffs[iab] * arrdata[PlasmaIdx::Fy1   + iab][ip];
                        ux  += ab5_coeffs[iab] * arrdata[PlasmaIdx::Fux1  + iab][ip];
                        uy  += ab5_coeffs[iab] * arrdata[PlasmaIdx::Fuy1  + iab][ip];
                        psi += ab5_coeffs[iab] * arrdata[PlasmaIdx::Fpsi1 + iab][ip];
                    }

                    if (setPositionEnforceBC(ip, xp, yp)) return;

                    if (!temp_slice) {
                        // update values of the last non temp slice
                        // the next push always starts from these
                        ux_half_step[ip] = ux;
                        uy_half_step[ip] = uy;
                        psi_half_step[ip] = psi;
                        x_prev[ip] = xp;
                        y_prev[ip] = yp;
                    }

                    uxp[ip] = ux;
                    uyp[ip] = uy;
                    psip[ip] = psi;
#endif
                });
        }

#ifdef HIPACE_USE_AB5_PUSH
        if (!temp_slice) {
            auto& rd = soa.GetRealData();

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
