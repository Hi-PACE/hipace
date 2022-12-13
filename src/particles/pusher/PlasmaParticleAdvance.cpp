/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaParticleAdvance.H"

#include "particles/plasma/PlasmaParticleContainer.H"
#include "GetDomainLev.H"
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
                        amrex::Geometry const& gm, const bool temp_slice, int const lev,
                        PlasmaBins& bins, const MultiLaser& multi_laser)
{
    HIPACE_PROFILE("AdvancePlasmaParticles()");
    using namespace amrex::literals;

    // only push plasma particles on their according MR level
    if (plasma.m_level != lev) return;

    const bool do_tiling = Hipace::m_do_tiling;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
    const PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract field array from FabArray
        const amrex::FArrayBox& slice_fab = fields.getSlices(lev, WhichSlice::This)[pti];
        Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
        const int exmby_comp = Comps[WhichSlice::This]["ExmBy"];
        const int eypbx_comp = Comps[WhichSlice::This]["EypBx"];
        const int ez_comp = Comps[WhichSlice::This]["Ez"];
        const int bx_comp = Comps[WhichSlice::This]["Bx"];
        const int by_comp = Comps[WhichSlice::This]["By"];
        const int bz_comp = Comps[WhichSlice::This]["Bz"];

        // extract the laser Fields
        const bool use_laser = multi_laser.m_use_laser;
        const amrex::MultiFab& a_mf = multi_laser.getSlices(WhichLaserSlice::n00j00);

        // Extract field array from MultiFab
        Array3<const amrex::Real> const& a_arr = use_laser ?
            a_mf[pti].const_array() : amrex::Array4<const amrex::Real>();

        const amrex::Real dx_inv = 1._rt/dx[0];
        const amrex::Real dy_inv = 1._rt/dx[1];

        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, gm, slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, slice_fab.box());

        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        // loading the data
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_half_step = soa.GetRealData(PlasmaIdx::ux_half_step).data();
        amrex::Real * const uy_half_step = soa.GetRealData(PlasmaIdx::uy_half_step).data();
        amrex::Real * const psi_inv_half_step =soa.GetRealData(PlasmaIdx::psi_inv_half_step).data();
        int * const ion_lev = plasma.m_can_ionize ? soa.GetIntData(PlasmaIdx::ion_lev).data()
                                                  : nullptr;

        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto setPositionEnforceBC = EnforceBCandSetPos<PTileType>(
            pti.GetParticleTile(), GetDomainLev(gm, pti.tilebox(), 1, lev),
            GetDomainLev(gm, pti.tilebox(), 0, lev), gm.isPeriodicArray());
        const amrex::Real dz = dx[2];

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
                    if (setPositionEnforceBC.m_structs[ip].id() < 0) return;

                    amrex::Real xp = x_prev[ip];
                    amrex::Real yp = y_prev[ip];

                    // define field at particle position reals
                    amrex::Real ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                    amrex::Real Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;
                    amrex::Real Aabssqp = 0._rt, AabssqDxp = 0._rt, AabssqDyp = 0._rt;

                    doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp,
                            Bzp, slice_arr, exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp,
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

                    constexpr int nsub = 4;
                    const amrex::Real sdz = dz/nsub;

                    amrex::Real ux = ux_half_step[ip];
                    amrex::Real uy = uy_half_step[ip];
                    amrex::Real psi_inv = psi_inv_half_step[ip];

                    amrex::Real ux_mid = 0._rt;
                    amrex::Real uy_mid = 0._rt;
                    amrex::Real psi_inv_mid = 0._rt;

                    for (int isub=0; isub<nsub; ++isub) {

                        auto [dz_ux, dz_uy, dz_psi_inv] = PlasmaMomentumPush(
                            ux, uy, psi_inv, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                        const DualNumber ux_dual{ux, dz_ux};
                        const DualNumber uy_dual{uy, dz_uy};
                        const DualNumber psi_inv_dual{psi_inv, dz_psi_inv};

                        auto [dz_ux_dual, dz_uy_dual, dz_psi_inv_dual] = PlasmaMomentumPush(
                            ux_dual, uy_dual, psi_inv_dual, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                        if (isub==(nsub/2)) {
                            ux_mid = ux;
                            uy_mid = uy;
                            psi_inv_mid = psi_inv;
                        }

                        ux += sdz*dz_ux + 0.5_rt*sdz*sdz*dz_ux_dual.epsilon;
                        uy += sdz*dz_uy + 0.5_rt*sdz*sdz*dz_uy_dual.epsilon;
                        psi_inv += sdz*dz_psi_inv + 0.5_rt*sdz*sdz*dz_psi_inv_dual.epsilon;

                    }

                    xp += dz*clight_inv*(ux * psi_inv);
                    yp += dz*clight_inv*(uy * psi_inv);

                    if (setPositionEnforceBC(ip, xp, yp)) return;

                    if (!temp_slice) {
                        ux_half_step[ip] = ux;
                        uy_half_step[ip] = uy;
                        psi_inv_half_step[ip] = psi_inv;
                        x_prev[ip] = xp;
                        y_prev[ip] = yp;
                    }

                    ux = ux_mid;
                    uy = uy_mid;
                    psi_inv = psi_inv_mid;

                    ExmByp = 0._rt;
                    EypBxp = 0._rt;
                    Ezp = 0._rt;
                    Bxp = 0._rt;
                    Byp = 0._rt;
                    Bzp = 0._rt;
                    Aabssqp = 0._rt;
                    AabssqDxp = 0._rt;
                    AabssqDyp = 0._rt;

                    doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp,
                            Bzp, slice_arr, exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp,
                            bz_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                    if (use_laser) {
                        doLaserGatherShapeN<depos_order.value>(xp, yp, Aabssqp, AabssqDxp,
                            AabssqDyp, a_arr, dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                    }
                    Bxp *= clight;
                    Byp *= clight;
                    Aabssqp *= 0.5_rt; // TODO: fix units of Aabssqp
                    AabssqDxp *= 0.25_rt * me_clight_mass_ratio;
                    AabssqDyp *= 0.25_rt * me_clight_mass_ratio;

                    for (int isub=0; isub<nsub; ++isub) {

                        auto [dz_ux, dz_uy, dz_psi_inv] = PlasmaMomentumPush(
                            ux, uy, psi_inv, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                        const DualNumber ux_dual{ux, dz_ux};
                        const DualNumber uy_dual{uy, dz_uy};
                        const DualNumber psi_inv_dual{psi_inv, dz_psi_inv};

                        auto [dz_ux_dual, dz_uy_dual, dz_psi_inv_dual] = PlasmaMomentumPush(
                            ux_dual, uy_dual, psi_inv_dual, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                            Aabssqp, AabssqDxp, AabssqDyp, clight_inv, q_mass_clight_ratio);

                        ux += sdz*dz_ux + 0.5_rt*sdz*sdz*dz_ux_dual.epsilon;
                        uy += sdz*dz_uy + 0.5_rt*sdz*sdz*dz_uy_dual.epsilon;
                        psi_inv += sdz*dz_psi_inv + 0.5_rt*sdz*sdz*dz_psi_inv_dual.epsilon;

                    }
                    uxp[ip] = ux;
                    uyp[ip] = uy;
                    psip[ip] = 1._rt/psi_inv;
                });
        }
    }
}

void
ResetPlasmaParticles (PlasmaParticleContainer& plasma, int const lev)
{
    HIPACE_PROFILE("ResetPlasmaParticles()");

    using namespace amrex::literals;
    const PhysConst phys_const = get_phys_const();

    const amrex::RealVect u_mean = plasma.GetUMean();
    const amrex::RealVect u_std = plasma.GetUStd();
    const int init_ion_lev = plasma.m_init_ion_lev;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        unsigned long num_initial_particles = plasma.m_init_num_par[pti.tileIndex()];
        pti.GetParticleTile().resize(num_initial_particles);

        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_half_step = soa.GetRealData(PlasmaIdx::ux_half_step).data();
        amrex::Real * const uy_half_step = soa.GetRealData(PlasmaIdx::uy_half_step).data();
        amrex::Real * const psi_inv_half_step =soa.GetRealData(PlasmaIdx::psi_inv_half_step).data();
        amrex::Real * const x0 = soa.GetRealData(PlasmaIdx::x0).data();
        amrex::Real * const y0 = soa.GetRealData(PlasmaIdx::y0).data();
        amrex::Real * const w = soa.GetRealData(PlasmaIdx::w).data();
        amrex::Real * const w0 = soa.GetRealData(PlasmaIdx::w0).data();
        int * const ion_lev = soa.GetIntData(PlasmaIdx::ion_lev).data();

        const auto GetPosition =
            GetParticlePosition<PlasmaParticleContainer::ParticleTileType>(pti.GetParticleTile());
        const auto SetPosition =
            SetParticlePosition<PlasmaParticleContainer::ParticleTileType>(pti.GetParticleTile());

        plasma.UpdateDensityFunction();
        auto density_func = plasma.m_density_func;
        const amrex::Real c_t = phys_const.c * Hipace::m_physical_time;

        amrex::ParallelForRNG(
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip, const amrex::RandomEngine& engine) {

                amrex::ParticleReal xp, yp, zp;
                int pid;
                GetPosition(ip, xp, yp, zp, pid);

                amrex::Real u[3] = {0._rt,0._rt,0._rt};
                ParticleUtil::get_gaussian_random_momentum(u, u_mean, u_std, engine);

                SetPosition(ip, x0[ip], y0[ip], zp, std::abs(pid));
                w[ip] = w0[ip] * density_func(x0[ip], y0[ip], c_t);
                uxp[ip] = u[0]*phys_const.c;
                uyp[ip] = u[1]*phys_const.c;
                psip[ip] = std::sqrt(1._rt + u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - u[2];
                x_prev[ip] = x0[ip];
                y_prev[ip] = y0[ip];
                ux_half_step[ip] = u[0]*phys_const.c;
                uy_half_step[ip] = u[1]*phys_const.c;
                psi_inv_half_step[ip] = 1._rt/psip[ip];
                ion_lev[ip] = init_ion_lev;
            }
            );
    }
}
