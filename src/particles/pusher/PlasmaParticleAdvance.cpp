/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaParticleAdvance.H"

#include "particles/PlasmaParticleContainer.H"
#include "GetDomainLev.H"
#include "FieldGather.H"
#include "PushPlasmaParticles.H"
#include "UpdateForceTerms.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"
#include "utils/HipaceProfilerWrapper.H"
#include "particles/ParticleUtil.H"

#include <string>

void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, Fields & fields,
                        amrex::Geometry const& gm, const bool temp_slice, const bool do_push,
                        const bool do_update, const bool do_shift, int const lev,
                        PlasmaBins& bins)
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
        // Extract the fields
        const amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
        const amrex::MultiFab exmby(S, amrex::make_alias, Comps[WhichSlice::This]["ExmBy"], 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, Comps[WhichSlice::This]["EypBx"], 1);
        const amrex::MultiFab ez(S, amrex::make_alias, Comps[WhichSlice::This]["Ez"], 1);
        const amrex::MultiFab bx(S, amrex::make_alias, Comps[WhichSlice::This]["Bx"], 1);
        const amrex::MultiFab by(S, amrex::make_alias, Comps[WhichSlice::This]["By"], 1);
        const amrex::MultiFab bz(S, amrex::make_alias, Comps[WhichSlice::This]["Bz"], 1);
        // Extract FabArray for this box
        const amrex::FArrayBox& exmby_fab = exmby[pti];
        const amrex::FArrayBox& eypbx_fab = eypbx[pti];
        const amrex::FArrayBox& ez_fab = ez[pti];
        const amrex::FArrayBox& bx_fab = bx[pti];
        const amrex::FArrayBox& by_fab = by[pti];
        const amrex::FArrayBox& bz_fab = bz[pti];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, gm, ez_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, gm, ez_fab.box());
        amrex::Real const z_pos_offset = GetPosOffset(2, gm, ez_fab.box());

        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        if (do_shift)
        {
            ShiftForceTerms(soa);
        }

        // loading the data
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        const amrex::Real * const const_of_motionp = soa.GetRealData(
                                                            PlasmaIdx::const_of_motion).data();
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

        const int depos_order_xy = Hipace::m_depos_order_xy;
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
                num_particles,
                [=] AMREX_GPU_DEVICE (long idx) {
                    const int ip = do_tiling ? indices[offsets[itile]+idx] : idx;
                    amrex::ParticleReal xp, yp, zp;
                    int pid;
                    getPosition(ip, xp, yp, zp, pid);

                    if (pid < 0) return;

                    // define field at particle position reals
                    amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                    amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                    if (do_update)
                    {
                        // field gather for a single particle
                        doGatherShapeN(xp, yp, 0 /* zp not used */,
                                       ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                                       exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                       dx_arr, x_pos_offset, y_pos_offset, z_pos_offset,
                                       depos_order_xy, 0);
                        // update force terms for a single particle
                        const amrex::Real q = can_ionize ? ion_lev[ip] * charge : charge;
                        const amrex::Real psi_factor = phys_const.q_e/(phys_const.m_e*phys_const.c*phys_const.c);
                        UpdateForceTerms(uxp[ip], uyp[ip], psi_factor*psip[ip], const_of_motionp[ip],
                                         ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, Fx1[ip], Fy1[ip],
                                         Fux1[ip], Fuy1[ip], Fpsi1[ip], clightsq, phys_const, q, mass);
                    }

                    if (do_push)
                    {
                        // push a single particle
                        PlasmaParticlePush(xp, yp, zp, uxp[ip], uyp[ip], psip[ip], x_prev[ip],
                                           y_prev[ip], ux_temp[ip], uy_temp[ip], psi_temp[ip],
                                           Fx1[ip], Fy1[ip], Fux1[ip], Fuy1[ip], Fpsi1[ip],
                                           Fx2[ip], Fy2[ip], Fux2[ip], Fuy2[ip], Fpsi2[ip],
                                           Fx3[ip], Fy3[ip], Fux3[ip], Fuy3[ip], Fpsi3[ip],
                                           Fx4[ip], Fy4[ip], Fux4[ip], Fuy4[ip], Fpsi4[ip],
                                           Fx5[ip], Fy5[ip], Fux5[ip], Fuy5[ip], Fpsi5[ip],
                                           dz, temp_slice, ip, SetPosition, enforceBC );
                    }
                    return;
                }
                );
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
        amrex::Real * const const_of_motionp = soa.GetRealData(PlasmaIdx::const_of_motion).data();
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

                    amrex::Real u[3] = {0.,0.,0.};
                    ParticleUtil::get_gaussian_random_momentum(u, u_mean, u_std, engine);

                    SetPosition(ip, x0[ip], y0[ip], zp, std::abs(pid));
                    w[ip] = w0[ip] * density_func(x0[ip], y0[ip], c_t);
                    uxp[ip] = u[0]*phys_const.c;
                    uyp[ip] = u[1]*phys_const.c;
                    psip[ip] = 0._rt;
                    x_prev[ip] = 0._rt;
                    y_prev[ip] = 0._rt;
                    ux_temp[ip] = 0._rt;
                    uy_temp[ip] = 0._rt;
                    psi_temp[ip] = 0._rt;
                    Fx1[ip] = 0._rt;
                    Fy1[ip] = 0._rt;
                    Fux1[ip] = 0._rt;
                    Fuy1[ip] = 0._rt;
                    Fpsi1[ip] = 0._rt;
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
                    const_of_motionp[ip]  = sqrt(1. + u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - u[2];
                }
            }
            );
    }
}
