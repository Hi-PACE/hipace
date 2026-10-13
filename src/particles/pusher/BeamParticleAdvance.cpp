
/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BeamParticleAdvance.H"
#include "ExternalFields.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/Constants.H"
#include "GetAndSetPosition.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"
#include "utils/OMPUtil.H"


void
AdvanceBeamParticlesSlice (
    BeamParticleContainer& beam, const Fields& fields, amrex::Vector<amrex::Geometry> const& gm,
    const int slice, int const current_N_level)
{
    HIPACE_PROFILE("AdvanceBeamParticlesSlice()");
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    const bool do_z_push = beam.m_do_z_push;
    const int n_subcycles = beam.m_n_subcycles;
    const bool radiation_reaction = beam.m_do_radiation_reaction;
    const amrex::Real time = Hipace::GetInstance().m_physical_time;
    const amrex::Real dt = Hipace::GetInstance().m_dt / n_subcycles;
    const bool spin_tracking = beam.m_do_spin_tracking;
    const amrex::Real spin_anom = beam.m_spin_anom;

    const int psi_comp = Comps[WhichSlice::This]["Psi"];
    const int ez_comp = Comps[WhichSlice::This]["Ez"];
    const int bx_comp = Comps[WhichSlice::This]["Bx"];
    const int by_comp = Comps[WhichSlice::This]["By"];
    const int bz_comp = Comps[WhichSlice::This]["Bz"];

    const bool do_ez_inzerp = (Hipace::m_depos_order_z == 2);
    const int ez_comp_prev = do_ez_inzerp ? Comps[WhichSlice::Previous]["Ez"] : -1;
    const int ez_comp_next = do_ez_inzerp ? Comps[WhichSlice::Next]["Ez"] : -1;

    const int lev0_idx = 0;
    const int lev1_idx = std::min(1, current_N_level-1);
    const int lev2_idx = std::min(2, current_N_level-1);

    // Extract field array from FabArrays in MultiFabs.
    // (because there is currently no transverse parallelization, the index
    // we want in the slice multifab is always 0. Fix later.
    const amrex::FArrayBox& slice_fab_lev0 = fields.getSlices(lev0_idx)[0];
    const amrex::FArrayBox& slice_fab_lev1 = fields.getSlices(lev1_idx)[0];
    const amrex::FArrayBox& slice_fab_lev2 = fields.getSlices(lev2_idx)[0];

    const MRLevelData level0data {
        slice_fab_lev0.const_array(),
        gm[lev0_idx].InvCellSize(0), gm[lev0_idx].InvCellSize(1),
        GetPosOffset(0, gm[lev0_idx], slice_fab_lev0.box()),
        GetPosOffset(1, gm[lev0_idx], slice_fab_lev0.box())
    };

    const MRLevelData level1data {
        slice_fab_lev1.const_array(),
        gm[lev1_idx].InvCellSize(0), gm[lev1_idx].InvCellSize(1),
        GetPosOffset(0, gm[lev1_idx], slice_fab_lev1.box()),
        GetPosOffset(1, gm[lev1_idx], slice_fab_lev1.box())
    };

    const MRLevelData level2data {
        slice_fab_lev2.const_array(),
        gm[lev2_idx].InvCellSize(0), gm[lev2_idx].InvCellSize(1),
        GetPosOffset(0, gm[lev2_idx], slice_fab_lev2.box()),
        GetPosOffset(1, gm[lev2_idx], slice_fab_lev2.box())
    };

    // z is the same for all levels
    amrex::Real const dz_inv = gm[lev0_idx].InvCellSize(2);

    const CheckDomainBounds lev1_bounds {gm[lev1_idx]};
    const CheckDomainBounds lev2_bounds {gm[lev2_idx]};

    // Extract particle properties
    const auto ptd = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();

    const auto enforceBC = EnforceBC();

    const amrex::Real clight = phys_const.c;
    const amrex::Real inv_clight = 1.0_rt/phys_const.c;
    const amrex::Real charge_mass_ratio = beam.m_charge / beam.m_mass;
    const amrex::Real min_z = gm[0].ProbLo(2) + (slice-gm[0].Domain().smallEnd(2))*gm[0].CellSize(2);
    const bool use_external_fields = beam.m_use_external_fields;
    const auto external_fields = beam.m_external_fields;

    // Radiation reaction constant
    amrex::Real rr_factor = (2.0_rt/3.0_rt) * PhysConstSI::r_e
        * charge_mass_ratio * charge_mass_ratio / PhysConstSI::c;
    if (Hipace::m_normalized_units && radiation_reaction) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_background_density_SI != 0,
            "For radiation reactions with normalized units, a background plasma density != 0 must "
            "be specified via 'hipace.background_density_SI'");
        rr_factor *= std::sqrt(static_cast<double>(Hipace::m_background_density_SI)
                / (PhysConstSI::ep0 * PhysConstSI::m_e)) * PhysConstSI::q_e;
    }

    // don't include slipped particles in count as they were already pushed
    Hipace::m_num_beam_particles_pushed += double(beam.getNumParticles(WhichBeamSlice::This));

    // Use OMP ParallelFor to use multiple threads when running on CPU
    omp::ParallelFor(
        amrex::TypeList<
            amrex::CompileTimeOptions<0, 1, 2, 3>,
            amrex::CompileTimeOptions<false, true>,
            amrex::CompileTimeOptions<false, true>
        >{}, {
            Hipace::m_depos_order_xy,
            use_external_fields,
            do_ez_inzerp
        },
        beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (int ip, auto depos_order, auto c_use_external_fields,
                              auto c_do_ez_inzerp) {

            if (!ptd.id(ip).is_valid()) return;

            // Load particle data
            amrex::Real xp = ptd.pos(0, ip);
            amrex::Real yp = ptd.pos(1, ip);
            amrex::Real zp = ptd.pos(2, ip);
            amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            int i = ptd.idata(BeamIdx::nsubcycles)[ip];

            amrex::RealVect spin {0._rt, 0._rt, 0._rt};
            if (spin_tracking) {
                spin[0] = ptd.rdata(BeamIdx::sx)[ip];
                spin[1] = ptd.rdata(BeamIdx::sy)[ip];
                spin[2] = ptd.rdata(BeamIdx::sz)[ip];
            }

            for (; i < n_subcycles; i++) {

                if (zp < min_z) {
                    // stop pushing particle if it is not on this slice anymore
                    break;
                }

                const amrex::Real gammap_inv = amrex::Math::rsqrt( 1._rt + ux*ux + uy*uy + uz*uz);

                // first we do half a step in x,y
                // This is not required in z, which is pushed in one step later
                xp += dt * clight * 0.5_rt * gammap_inv * ux;
                yp += dt * clight * 0.5_rt * gammap_inv * uy;

                if (enforceBC(ptd, ip, xp, yp, ux, uy)) return;

                // Load field data from highest available MR level
                MRLevelData level_data = level0data; // level 0
                if (current_N_level > 2 && lev2_bounds.contains(xp, yp)) {
                    level_data = level2data; // level 2
                } else if (current_N_level > 1 && lev1_bounds.contains(xp, yp)) {
                    level_data = level1data; // level 1
                }
                const auto [slice_arr, dx_inv, dy_inv, x_pos_offset, y_pos_offset] = level_data;

                // define field at particle position reals
                amrex::Real ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::Real Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                // field gather for a single particle
                doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                    slice_arr, psi_comp, ez_comp, bx_comp, by_comp, bz_comp,
                    dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                if (c_do_ez_inzerp.value) {
                    // Update Ez
                    InterpolateEzInZ<depos_order.value>(Ezp,
                        xp, yp, zp, x_pos_offset, y_pos_offset, min_z, dx_inv, dy_inv, dz_inv,
                        ez_comp_prev, ez_comp_next, slice_arr);
                }

                if (c_use_external_fields.value) {
                    // Update ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp
                    ApplyExternalField(xp, yp, zp, time, clight, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                        external_fields);
                }

                ExmByp *= inv_clight;
                EypBxp *= inv_clight;
                Ezp *= inv_clight;

                // use intermediate fields to calculate next (n+1) transverse momenta
                amrex::Real ux_next = ux + dt * charge_mass_ratio
                    * ( ExmByp + ( 1._rt - uz * gammap_inv ) * Byp + uy * gammap_inv * Bzp);
                amrex::Real uy_next = uy + dt * charge_mass_ratio
                    * ( EypBxp - ( 1._rt - uz * gammap_inv ) * Bxp - ux * gammap_inv * Bzp);

                // Now computing new longitudinal momentum
                const amrex::Real ux_intermediate = ( ux_next + ux ) * 0.5_rt;
                const amrex::Real uy_intermediate = ( uy_next + uy ) * 0.5_rt;
                const amrex::Real uz_intermediate = uz
                    + dt * 0.5_rt * charge_mass_ratio * Ezp;

                const amrex::Real gamma_intermediate_inv = amrex::Math::rsqrt( 1._rt
                    + ux_intermediate*ux_intermediate
                    + uy_intermediate*uy_intermediate
                    + uz_intermediate*uz_intermediate);

                if (spin_tracking) {
                    // Update spin
                    PushSpin(spin,
                        ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, ux_intermediate, uy_intermediate,
                        uz_intermediate, gamma_intermediate_inv, charge_mass_ratio, dt, spin_anom);
                }

                amrex::Real uz_next = uz + dt * charge_mass_ratio
                    * ( Ezp + ( ux_intermediate * Byp - uy_intermediate * Bxp )
                    * gamma_intermediate_inv );

                if (radiation_reaction) {
                    // Update ux_next, uy_next, uz_next
                    ApplyRadiationReaction(ux_next, uy_next, uz_next,
                        ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, ux_intermediate,
                        uy_intermediate, uz_intermediate, gamma_intermediate_inv,
                        dt, rr_factor
                    );
                }

                /* computing next gamma value */
                const amrex::Real gamma_next_inv = amrex::Math::rsqrt( 1._rt
                    + ux_next*ux_next
                    + uy_next*uy_next
                    + uz_next*uz_next);

                /*
                 * computing positions and setting momenta for the next timestep
                 *(n+1)
                 * The longitudinal position is updated here as well, but in
                 * first-order (i.e. without the intermediary half-step) using
                 * a simple Galilean transformation
                 */
                xp += dt * clight * 0.5_rt * gamma_next_inv * ux_next;
                yp += dt * clight * 0.5_rt * gamma_next_inv * uy_next;
                if (do_z_push) {
                    zp += dt * clight * ( uz_next * gamma_next_inv - 1._rt );
                }
                ux = ux_next;
                uy = uy_next;
                uz = uz_next;

            } // end for loop over n_subcycles
            if (enforceBC(ptd, ip, xp, yp, ux, uy)) return;

            // Store particle data
            ptd.pos(0, ip) = xp;
            ptd.pos(1, ip) = yp;
            ptd.pos(2, ip) = zp;
            ptd.idata(BeamIdx::nsubcycles)[ip] = i;
            ptd.rdata(BeamIdx::ux)[ip] = ux;
            ptd.rdata(BeamIdx::uy)[ip] = uy;
            ptd.rdata(BeamIdx::uz)[ip] = uz;

            if (spin_tracking) {
                ptd.rdata(BeamIdx::sx)[ip] = spin[0];
                ptd.rdata(BeamIdx::sy)[ip] = spin[1];
                ptd.rdata(BeamIdx::sz)[ip] = spin[2];
            }
        });
}
