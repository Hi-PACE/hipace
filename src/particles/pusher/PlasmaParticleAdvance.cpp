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
#include "UpdateForceTerms.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"
#include "utils/DualNumbers.H"
#include "particles/particles_utils/ParticleUtil.H"

#include <string>

void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, const Fields & fields,
                        amrex::Geometry const& gm, const bool temp_slice, const bool do_push,
                        const bool do_update, const bool do_shift, int const lev,
                        PlasmaBins& bins, const MultiLaser& multi_laser)
{
    std::string str = "UpdateForcePushParticles_Plasma(    )";
    if (temp_slice) str.at(32) = 't';
    if (do_push) str.at(33) = 'p';
    if (do_update) str.at(34) = 'u';
    if (do_shift) str.at(35) = 's';
    HIPACE_PROFILE(str);
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

        if (do_shift)
        {
            //ShiftForceTerms(soa);
        }

        // loading the data
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_temp = soa.GetRealData(PlasmaIdx::ux_temp).data();
        amrex::Real * const uy_temp = soa.GetRealData(PlasmaIdx::uy_temp).data();
        amrex::Real * const psi_temp = soa.GetRealData(PlasmaIdx::psi_temp).data();

        amrex::Real * const Fx1 = soa.GetRealData(PlasmaIdx::Fx1).data();
        amrex::Real * const Fy1 = soa.GetRealData(PlasmaIdx::Fy1).data();
        amrex::Real * const Fux1 = soa.GetRealData(PlasmaIdx::Fux1).data();
        amrex::Real * const Fuy1 = soa.GetRealData(PlasmaIdx::Fuy1).data();
        amrex::Real * const Fpsi1 = soa.GetRealData(PlasmaIdx::Fpsi1).data();
        amrex::Real * const Fx2 = soa.GetRealData(PlasmaIdx::Fx2).data();
        amrex::Real * const Fy2 = soa.GetRealData(PlasmaIdx::Fy2).data();
        amrex::Real * const Fux2 = soa.GetRealData(PlasmaIdx::Fux2).data();
        amrex::Real * const Fuy2 = soa.GetRealData(PlasmaIdx::Fuy2).data();
        amrex::Real * const Fpsi2 = soa.GetRealData(PlasmaIdx::Fpsi2).data();
        amrex::Real * const Fx3 = soa.GetRealData(PlasmaIdx::Fx3).data();
        amrex::Real * const Fy3 = soa.GetRealData(PlasmaIdx::Fy3).data();
        amrex::Real * const Fux3 = soa.GetRealData(PlasmaIdx::Fux3).data();
        amrex::Real * const Fuy3 = soa.GetRealData(PlasmaIdx::Fuy3).data();
        amrex::Real * const Fpsi3 = soa.GetRealData(PlasmaIdx::Fpsi3).data();
        amrex::Real * const Fx4 = soa.GetRealData(PlasmaIdx::Fx4).data();
        amrex::Real * const Fy4 = soa.GetRealData(PlasmaIdx::Fy4).data();
        amrex::Real * const Fux4 = soa.GetRealData(PlasmaIdx::Fux4).data();
        amrex::Real * const Fuy4 = soa.GetRealData(PlasmaIdx::Fuy4).data();
        amrex::Real * const Fpsi4 = soa.GetRealData(PlasmaIdx::Fpsi4).data();
        amrex::Real * const Fx5 = soa.GetRealData(PlasmaIdx::Fx5).data();
        amrex::Real * const Fy5 = soa.GetRealData(PlasmaIdx::Fy5).data();
        amrex::Real * const Fux5 = soa.GetRealData(PlasmaIdx::Fux5).data();
        amrex::Real * const Fuy5 = soa.GetRealData(PlasmaIdx::Fuy5).data();
        amrex::Real * const Fpsi5 = soa.GetRealData(PlasmaIdx::Fpsi5).data();
        int * const ion_lev = soa.GetIntData(PlasmaIdx::ion_lev).data();

        const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto getPosition = GetParticlePosition<PTileType>(pti.GetParticleTile());
        const auto SetPosition = SetParticlePosition<PTileType>(pti.GetParticleTile());
        const auto enforceBC = EnforceBC<PTileType>(
            pti.GetParticleTile(), GetDomainLev(gm, pti.tilebox(), 1, lev),
            GetDomainLev(gm, pti.tilebox(), 0, lev), gm.isPeriodicArray());
        const amrex::Real dz = dx[2];

        const amrex::Real charge = plasma.m_charge;
        const amrex::Real mass = plasma.m_mass;
        const bool can_ionize = plasma.m_can_ionize;

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
                    amrex::ParticleReal xp, yp, zp;
                    int pid;
                    getPosition(ip, xp, yp, zp, pid);

                    if (pid < 0) return;

                    // define field at particle position reals
                    amrex::Real ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                    amrex::Real Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;
                    amrex::Real Aabssqp = 0._rt, AabssqDxp = 0._rt, AabssqDyp = 0._rt;

                    doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp,
                            Bzp, slice_arr, exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp,
                            bz_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                    const amrex::Real q = can_ionize ? ion_lev[ip] * charge : charge;
                    amrex::Real charge_mass_ratio = q / mass;
                    const amrex::Real clight_inv = 1/phys_const.c;
                    const amrex::Real mass_inv = 1/mass;
                    Bxp *= phys_const.c;
                    Byp *= phys_const.c;
                    AabssqDxp *= 0.25_rt * phys_const.c * phys_const.m_e * mass_inv;
                    AabssqDyp *= 0.25_rt * phys_const.c * phys_const.m_e * mass_inv;
                    charge_mass_ratio *= clight_inv;

                    constexpr int nsub = 4;
                    const amrex::Real dz_sub = dz/nsub;

                    amrex::Real ux_n = Fux1[ip];
                    amrex::Real uy_n = Fuy1[ip];
                    amrex::Real psi_inv_n = Fpsi1[ip];

                    amrex::Real dz_ux_t = 0;
                    amrex::Real dz_uy_t = 0;
                    amrex::Real dz_psi_inv_t = 0;
                    DualNumber dz_ux_d;
                    DualNumber dz_uy_d;
                    DualNumber dz_psi_inv_d;


                    for (int isub=0; isub<nsub; ++isub) {

                        const amrex::Real ux_t = ux_n;
                        const amrex::Real uy_t = uy_n;
                        const amrex::Real psi_inv_t = psi_inv_n;

                        const amrex::Real gammap_psi = 0.5_rt*psi_inv_t*psi_inv_t*(
                                        1.0_rt
                                        + ux_t*ux_t*clightsq
                                        + uy_t*uy_t*clightsq // TODO: fix units of Aabssqp
                                        + 0.5_rt*Aabssqp)+0.5_rt;

                        dz_ux_t = (charge_mass_ratio * (gammap_psi * ExmByp +
                                Byp + ( uy_t * Bzp ) * psi_inv_t) - AabssqDxp * psi_inv_t);

                        dz_uy_t = (charge_mass_ratio * ( gammap_psi * EypBxp -
                                Bxp - ( ux_t * Bzp ) * psi_inv_t) - AabssqDyp * psi_inv_t);

                        dz_psi_inv_t = psi_inv_t*psi_inv_t*(-charge_mass_ratio * clight_inv *
                                (( ux_t*ExmByp + uy_t*EypBxp ) * clight_inv * psi_inv_t - Ezp ));

                        const DualNumber ux_d{ux_t, dz_ux_t};
                        const DualNumber uy_d{uy_t, dz_uy_t};
                        const DualNumber psi_inv_d{psi_inv_t, dz_psi_inv_t};

                        const DualNumber gammap_psi_d = 0.5_rt*psi_inv_d*psi_inv_d*(
                                        1.0_rt
                                        + ux_d*ux_d*clightsq
                                        + uy_d*uy_d*clightsq // TODO: fix units of Aabssqp
                                        + 0.5_rt*Aabssqp)+0.5_rt;

                        dz_ux_d = (charge_mass_ratio * (gammap_psi_d * ExmByp +
                                Byp + ( uy_d * Bzp ) * psi_inv_d) - AabssqDxp * psi_inv_d);

                        dz_uy_d = (charge_mass_ratio * ( gammap_psi_d * EypBxp -
                                Bxp - ( ux_d * Bzp ) * psi_inv_d) - AabssqDyp * psi_inv_d);

                        dz_psi_inv_d = psi_inv_d*psi_inv_d*(-charge_mass_ratio * clight_inv *
                                (( ux_d*ExmByp + uy_d*EypBxp ) * clight_inv * psi_inv_d - Ezp ));

                        ux_n += dz_sub*dz_ux_t + 0.5_rt*dz_sub*dz_sub*dz_ux_d.epsilon;
                        uy_n += dz_sub*dz_uy_t + 0.5_rt*dz_sub*dz_sub*dz_uy_d.epsilon;
                        psi_inv_n += dz_sub*dz_psi_inv_t + 0.5_rt*dz_sub*dz_sub*dz_psi_inv_d.epsilon;

                    }

                    Fux1[ip] = ux_n;
                    Fuy1[ip] = uy_n;
                    Fpsi1[ip] = psi_inv_n;

                    xp += dz*clight_inv*(ux_n * psi_inv_n);
                    yp += dz*clight_inv*(uy_n * psi_inv_n);

                    SetPosition(ip, xp, yp, zp);
                    if (enforceBC(ip)) return;

                    x_prev[ip] = xp;
                    y_prev[ip] = yp;

                    ExmByp *= -0.5_rt;
                    EypBxp *= -0.5_rt;
                    Ezp *= -0.5_rt;
                    Bxp *= -0.5_rt;
                    Byp *= -0.5_rt;
                    Bzp *= -0.5_rt;
                    Aabssqp *= -0.5_rt;
                    AabssqDxp *= -0.5_rt;
                    AabssqDyp *= -0.5_rt;

                    doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp,
                            Bzp, slice_arr, exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp,
                            bz_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                    ExmByp *= 2._rt;
                    EypBxp *= 2._rt;
                    Ezp *= 2._rt;
                    Bxp *= 2._rt;
                    Byp *= 2._rt;
                    Bzp *= 2._rt;
                    Aabssqp *= 2._rt;
                    AabssqDxp *= 2._rt;
                    AabssqDyp *= 2._rt;

                    for (int isub=0; isub<(nsub/2); ++isub) {

                        const amrex::Real ux_t = ux_n;
                        const amrex::Real uy_t = uy_n;
                        const amrex::Real psi_inv_t = psi_inv_n;

                        const amrex::Real gammap_psi = 0.5_rt*psi_inv_t*psi_inv_t*(
                                        1.0_rt
                                        + ux_t*ux_t*clightsq
                                        + uy_t*uy_t*clightsq // TODO: fix units of Aabssqp
                                        + 0.5_rt*Aabssqp)+0.5_rt;

                        dz_ux_t = (charge_mass_ratio * (gammap_psi * ExmByp +
                                Byp + ( uy_t * Bzp ) * psi_inv_t) - AabssqDxp * psi_inv_t);

                        dz_uy_t = (charge_mass_ratio * ( gammap_psi * EypBxp -
                                Bxp - ( ux_t * Bzp ) * psi_inv_t) - AabssqDyp * psi_inv_t);

                        dz_psi_inv_t = psi_inv_t*psi_inv_t*(-charge_mass_ratio * clight_inv *
                                (( ux_t*ExmByp + uy_t*EypBxp ) * clight_inv * psi_inv_t - Ezp ));

                        const DualNumber ux_d{ux_t, dz_ux_t};
                        const DualNumber uy_d{uy_t, dz_uy_t};
                        const DualNumber psi_inv_d{psi_inv_t, dz_psi_inv_t};

                        const DualNumber gammap_psi_d = 0.5_rt*psi_inv_d*psi_inv_d*(
                                        1.0_rt
                                        + ux_d*ux_d*clightsq
                                        + uy_d*uy_d*clightsq // TODO: fix units of Aabssqp
                                        + 0.5_rt*Aabssqp)+0.5_rt;

                        dz_ux_d = (charge_mass_ratio * (gammap_psi_d * ExmByp +
                                Byp + ( uy_d * Bzp ) * psi_inv_d) - AabssqDxp * psi_inv_d);

                        dz_uy_d = (charge_mass_ratio * ( gammap_psi_d * EypBxp -
                                Bxp - ( ux_d * Bzp ) * psi_inv_d) - AabssqDyp * psi_inv_d);

                        dz_psi_inv_d = psi_inv_d*psi_inv_d*(-charge_mass_ratio * clight_inv *
                                (( ux_d*ExmByp + uy_d*EypBxp ) * clight_inv * psi_inv_d - Ezp ));

                        ux_n += dz_sub*dz_ux_t + 0.5_rt*dz_sub*dz_sub*dz_ux_d.epsilon;
                        uy_n += dz_sub*dz_uy_t + 0.5_rt*dz_sub*dz_sub*dz_uy_d.epsilon;
                        psi_inv_n += dz_sub*dz_psi_inv_t + 0.5_rt*dz_sub*dz_sub*dz_psi_inv_d.epsilon;

                    }
                    uxp[ip] = ux_n;
                    uyp[ip] = uy_n;
                    psip[ip] = 1._rt/psi_inv_n;
                });
        }
    }
}

void
ResetPlasmaParticles (PlasmaParticleContainer& plasma, int const lev, const bool initial)
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
        if(initial) {
            // reset size to initial value
            unsigned long num_initial_particles = plasma.m_init_num_par[pti.tileIndex()];
            pti.GetParticleTile().resize(num_initial_particles);
        }

        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_temp = soa.GetRealData(PlasmaIdx::ux_temp).data();
        amrex::Real * const uy_temp = soa.GetRealData(PlasmaIdx::uy_temp).data();
        amrex::Real * const psi_temp = soa.GetRealData(PlasmaIdx::psi_temp).data();
        amrex::Real * const Fx1 = soa.GetRealData(PlasmaIdx::Fx1).data();
        amrex::Real * const Fy1 = soa.GetRealData(PlasmaIdx::Fy1).data();
        amrex::Real * const Fux1 = soa.GetRealData(PlasmaIdx::Fux1).data();
        amrex::Real * const Fuy1 = soa.GetRealData(PlasmaIdx::Fuy1).data();
        amrex::Real * const Fpsi1 = soa.GetRealData(PlasmaIdx::Fpsi1).data();
        amrex::Real * const Fx2 = soa.GetRealData(PlasmaIdx::Fx2).data();
        amrex::Real * const Fy2 = soa.GetRealData(PlasmaIdx::Fy2).data();
        amrex::Real * const Fux2 = soa.GetRealData(PlasmaIdx::Fux2).data();
        amrex::Real * const Fuy2 = soa.GetRealData(PlasmaIdx::Fuy2).data();
        amrex::Real * const Fpsi2 = soa.GetRealData(PlasmaIdx::Fpsi2).data();
        amrex::Real * const Fx3 = soa.GetRealData(PlasmaIdx::Fx3).data();
        amrex::Real * const Fy3 = soa.GetRealData(PlasmaIdx::Fy3).data();
        amrex::Real * const Fux3 = soa.GetRealData(PlasmaIdx::Fux3).data();
        amrex::Real * const Fuy3 = soa.GetRealData(PlasmaIdx::Fuy3).data();
        amrex::Real * const Fpsi3 = soa.GetRealData(PlasmaIdx::Fpsi3).data();
        amrex::Real * const Fx4 = soa.GetRealData(PlasmaIdx::Fx4).data();
        amrex::Real * const Fy4 = soa.GetRealData(PlasmaIdx::Fy4).data();
        amrex::Real * const Fux4 = soa.GetRealData(PlasmaIdx::Fux4).data();
        amrex::Real * const Fuy4 = soa.GetRealData(PlasmaIdx::Fuy4).data();
        amrex::Real * const Fpsi4 = soa.GetRealData(PlasmaIdx::Fpsi4).data();
        amrex::Real * const Fx5 = soa.GetRealData(PlasmaIdx::Fx5).data();
        amrex::Real * const Fy5 = soa.GetRealData(PlasmaIdx::Fy5).data();
        amrex::Real * const Fux5 = soa.GetRealData(PlasmaIdx::Fux5).data();
        amrex::Real * const Fuy5 = soa.GetRealData(PlasmaIdx::Fuy5).data();
        amrex::Real * const Fpsi5 = soa.GetRealData(PlasmaIdx::Fpsi5).data();
        amrex::Real * const x0 = soa.GetRealData(PlasmaIdx::x0).data();
        amrex::Real * const y0 = soa.GetRealData(PlasmaIdx::y0).data();
        amrex::Real * const w = soa.GetRealData(PlasmaIdx::w).data();
        amrex::Real * const w0 = soa.GetRealData(PlasmaIdx::w0).data();
        int * const ion_lev = soa.GetIntData(PlasmaIdx::ion_lev).data();

        const auto GetPosition =
            GetParticlePosition<PlasmaParticleContainer::ParticleTileType>(pti.GetParticleTile());
        const auto SetPosition =
            SetParticlePosition<PlasmaParticleContainer::ParticleTileType>(pti.GetParticleTile());

        if (initial) plasma.UpdateDensityFunction();
        auto density_func = plasma.m_density_func;
        const amrex::Real c_t = phys_const.c * Hipace::m_physical_time;

        amrex::ParallelForRNG(
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip, const amrex::RandomEngine& engine) {

                amrex::ParticleReal xp, yp, zp;
                int pid;
                GetPosition(ip, xp, yp, zp, pid);
                if (initial == false){
                    SetPosition(ip, x_prev[ip], y_prev[ip], zp);
                } else {

                    amrex::Real u[3] = {0._rt,0._rt,0._rt};
                    ParticleUtil::get_gaussian_random_momentum(u, u_mean, u_std, engine);

                    SetPosition(ip, x0[ip], y0[ip], zp, std::abs(pid));
                    w[ip] = w0[ip] * density_func(x0[ip], y0[ip], c_t);
                    uxp[ip] = u[0]*phys_const.c;
                    uyp[ip] = u[1]*phys_const.c;
                    psip[ip] = sqrt(1._rt + u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - u[2];
                    x_prev[ip] = 0._rt;
                    y_prev[ip] = 0._rt;
                    ux_temp[ip] = 0._rt;
                    uy_temp[ip] = 0._rt;
                    psi_temp[ip] = 0._rt;
                    Fx1[ip] = 0._rt;
                    Fy1[ip] = 0._rt;
                    Fux1[ip] = u[0]*phys_const.c;
                    Fuy1[ip] = u[1]*phys_const.c;
                    Fpsi1[ip] = 1/(sqrt(1._rt + u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - u[2]);
                    Fx2[ip] = 0._rt;
                    Fy2[ip] = 0._rt;
                    Fux2[ip] = 0._rt;
                    Fuy2[ip] = 0._rt;
                    Fpsi2[ip] = 0._rt;
                    Fx3[ip] = 0._rt;
                    Fy3[ip] = 0._rt;
                    Fux3[ip] = 0._rt;
                    Fuy3[ip] = 0._rt;
                    Fpsi3[ip] = 0._rt;
                    Fx4[ip] = 0._rt;
                    Fy4[ip] = 0._rt;
                    Fux4[ip] = 0._rt;
                    Fuy4[ip] = 0._rt;
                    Fpsi4[ip] = 0._rt;
                    Fx5[ip] = 0._rt;
                    Fy5[ip] = 0._rt;
                    Fux5[ip] = 0._rt;
                    Fuy5[ip] = 0._rt;
                    Fpsi5[ip] = 0._rt;
                    ion_lev[ip] = init_ion_lev;
                }
            }
            );
    }
}
