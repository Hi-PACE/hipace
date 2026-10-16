/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "HelmholtzDeposition.H"
#include "DepositionUtil.H"
#include "particles/beam/BeamParticleContainer.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "helmholtz/Helmholtz.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"
#include <cmath>

void
HelmholtzDeposition (BeamParticleContainer& beam, Helmholtz& helmholtz,
                     const int which_beam_slice, amrex::Real time)
{
    HIPACE_PROFILE("HelmholtzDeposition()");

    using namespace amrex::literals;

    amrex::FArrayBox& isl_fab = helmholtz.getSlices()[0];
    const amrex::Geometry& gm = helmholtz.GetHelmholtzGeom();
    const CheckDomainBounds helmholtz_bounds {gm};
    bool mode_is_envelope = helmholtz.ModeIsEnvelope();


    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset = GetPosOffset(0, gm, gm.Domain());
    amrex::Real const y_pos_offset = GetPosOffset(1, gm, gm.Domain());

    PhysConst const phys_const = get_phys_const();

    if (mode_is_envelope) {
        bool in_undulator = false;
        for (int iu=0; iu<beam.m_nundulator; iu++) {
            const amrex::Real zprop = phys_const.c*time - beam.m_undulator_z[iu];
            const amrex::Real undulator_l = beam.m_undulator_nperiod[iu]*beam.m_undulator_period[0];
            if ((zprop >= 0 && zprop < undulator_l))
            {
                in_undulator = true;
            }
        }
        if (!in_undulator) return;
    }

    const amrex::Real ku = 2.*MathConst::pi/beam.m_undulator_period[0];
    const amrex::Real kr = helmholtz.getk0();
    const amrex::Real fcK = beam.m_undulator_fcK[0];

    // Extract box properties
    const amrex::Real dxi = gm.InvCellSize(0);
    const amrex::Real dyi = gm.InvCellSize(1);
    const amrex::Real dzi = gm.InvCellSize(2);
    amrex::Real invvol = dxi * dyi * dzi;

    if (Hipace::m_normalized_units) {
        const amrex::Geometry& lev0_geom = Hipace::GetInstance().m_3D_geom[0];
        invvol = lev0_geom.CellSize(0) * lev0_geom.CellSize(1) * dxi * dyi;
    }

    const amrex::Real q = beam.m_charge;

    int jx_slice {-1};
    int jz_slice {-1};
    int rho_slice {-1};
    if (which_beam_slice == WhichBeamSlice::Next) {
        jx_slice = WhichHelmholtzSlice::jx_n00jm1;
        jz_slice = WhichHelmholtzSlice::jz_n00jm1;
        rho_slice = WhichHelmholtzSlice::rho_n00jm1;
    } else {
        jx_slice = WhichHelmholtzSlice::jx_n00j00;
        jz_slice = WhichHelmholtzSlice::jz_n00j00;
        rho_slice = WhichHelmholtzSlice::rho_n00j00;
    }

    amrex::AnyCTO(
        // use compile-time options
        amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
        {Hipace::m_depos_order_xy},
        // call deposition function
        // The three functions passed as arguments to this lambda
        // are defined below as the next arguments.
        [&](auto is_valid, auto get_cell, auto deposit){
            constexpr auto ctos = deposit.GetOptions();
            constexpr int depos_order = ctos[0];
            constexpr int stencil_size = depos_order + 1;
            SharedMemoryDeposition<stencil_size, stencil_size, true>(
                beam.getNumParticles(which_beam_slice), is_valid, get_cell, deposit,
                isl_fab.array(), isl_fab.box(),
                beam.getBeamSlice(which_beam_slice).getParticleTileData(),
                amrex::GpuArray<int, 0>{},
                amrex::GpuArray<int, 3>{
                    jx_slice,
                    jz_slice,
                    rho_slice
                });
        },
        // is_valid
        // return whether the particle is valid and should deposit
        [=] AMREX_GPU_DEVICE (int ip, auto ptd, auto /*depos_order*/)
        {
            const amrex::Real xp = ptd.pos(0, ip);
            const amrex::Real yp = ptd.pos(1, ip);
            return ptd.id(ip).is_valid() && helmholtz_bounds.contains(xp, yp);
        },
        // get_cell
        // return the lowest cell index that the particle deposits into
        [=] AMREX_GPU_DEVICE (int ip, auto ptd, auto depos_order) -> amrex::IntVectND<2>
        {
            const amrex::Real xmid = (ptd.pos(0, ip) - x_pos_offset)*dxi;
            const amrex::Real ymid = (ptd.pos(1, ip) - y_pos_offset)*dyi;

            // --- Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order>(ymid, 0);
            auto [shape_x, i] = shape_factor<depos_order>(xmid, 0);

            return {i, j};
        },
        // deposit
        // deposit the charge / current of one particle
        [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                              Array3<amrex::Real> arr,
                              auto /*cache_idx*/, auto depos_idx,
                              auto depos_order) {
            // --- Get particle quantities
            const amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            const amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            const amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + ux*ux + uy*uy + uz*uz);
            const amrex::Real wq = q*ptd.rdata(BeamIdx::w)[ip]*invvol;
            const amrex::Real theta = (kr+ku)*ptd.pos(2, ip) + ku*phys_const.c*time;
            const amrex::Real wqg = wq * gaminv * q * PhysConstSI::mu0 / PhysConstSI::m_e;

            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = wq*ux*gaminv*phys_const.c;
            // const amrex::Real wqy = wq*uy*gaminv;
            const amrex::Real wqz = wq*uz*gaminv*phys_const.c;
            const amrex::Real wqrho = wq;

            const amrex::Real xmid = (ptd.pos(0, ip) - x_pos_offset)*dxi;
            const amrex::Real ymid = (ptd.pos(1, ip) - y_pos_offset)*dyi;

            // Deposit current into jx, jy, jz, rhomjz
            for (int iy=0; iy<=depos_order; iy++){
                for (int ix=0; ix<=depos_order; ix++){

                    // --- Compute shape factors
                    auto [shape_y, j] = shape_factor<depos_order>(ymid, iy);
                    auto [shape_x, i] = shape_factor<depos_order>(xmid, ix);

                    if (mode_is_envelope) {
                        // jx  array contains chi   =  e^2*mu0/me*ne/gamma_j
                        // jz  array contains Re(S) = fcK*sin(theta_j)*chi
                        // rho array contains Im(S) = fcK*cos(theta_j)*chi
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i, j, depos_idx[0]), shape_x * shape_y * wqg);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i, j, depos_idx[1]),
                            shape_x * shape_y * fcK * std::sin(theta) * wqg);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i, j, depos_idx[2]),
                            shape_x * shape_y * fcK * std::cos(theta) * wqg);
                    } else {
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i, j, depos_idx[0]), shape_x * shape_y * wqx);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i, j, depos_idx[1]), shape_x * shape_y * wqz);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i, j, depos_idx[2]), shape_x * shape_y * wqrho);
                    }
                }
            }
        });
}
