
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
    const amrex::Real background_density_SI = Hipace::m_background_density_SI;
    const bool normalized_units = Hipace::m_normalized_units;
    const bool spin_tracking = beam.m_do_spin_tracking;
    const amrex::Real spin_anom = beam.m_spin_anom;

    if (normalized_units && radiation_reaction) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(background_density_SI!=0,
            "For radiation reactions with normalized units, a background plasma density != 0 must "
            "be specified via 'hipace.background_density_SI'");
    }

    const int psi_comp = Comps[WhichSlice::This]["Psi"];
    const int ez_comp = Comps[WhichSlice::This]["Ez"];
    const int bx_comp = Comps[WhichSlice::This]["Bx"];
    const int by_comp = Comps[WhichSlice::This]["By"];
    const int bz_comp = Comps[WhichSlice::This]["Bz"];

    const int lev0_idx = 0;
    const int lev1_idx = std::min(1, current_N_level-1);
    const int lev2_idx = std::min(2, current_N_level-1);

    // Extract field array from FabArrays in MultiFabs.
    // (because there is currently no transverse parallelization, the index
    // we want in the slice multifab is always 0. Fix later.
    const amrex::FArrayBox& slice_fab_lev0 = fields.getSlices(lev0_idx)[0];
    const amrex::FArrayBox& slice_fab_lev1 = fields.getSlices(lev1_idx)[0];
    const amrex::FArrayBox& slice_fab_lev2 = fields.getSlices(lev2_idx)[0];

    Array3<const amrex::Real> const slice_arr_lev0 = slice_fab_lev0.const_array();
    Array3<const amrex::Real> const slice_arr_lev1 = slice_fab_lev1.const_array();
    Array3<const amrex::Real> const slice_arr_lev2 = slice_fab_lev2.const_array();

    // Extract properties associated with physical size of the box
    const amrex::Real dx_inv_lev0 = gm[lev0_idx].InvCellSize(0);
    const amrex::Real dx_inv_lev1 = gm[lev1_idx].InvCellSize(0);
    const amrex::Real dx_inv_lev2 = gm[lev2_idx].InvCellSize(0);

    amrex::Real const dy_inv_lev0 = gm[lev0_idx].InvCellSize(1);
    const amrex::Real dy_inv_lev1 = gm[lev1_idx].InvCellSize(1);
    const amrex::Real dy_inv_lev2 = gm[lev2_idx].InvCellSize(1);

    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset_lev0 = GetPosOffset(0, gm[lev0_idx], slice_fab_lev0.box());
    amrex::Real const x_pos_offset_lev1 = GetPosOffset(0, gm[lev1_idx], slice_fab_lev1.box());
    amrex::Real const x_pos_offset_lev2 = GetPosOffset(0, gm[lev2_idx], slice_fab_lev2.box());

    const amrex::Real y_pos_offset_lev0 = GetPosOffset(1, gm[lev0_idx], slice_fab_lev0.box());
    const amrex::Real y_pos_offset_lev1 = GetPosOffset(1, gm[lev1_idx], slice_fab_lev1.box());
    const amrex::Real y_pos_offset_lev2 = GetPosOffset(1, gm[lev2_idx], slice_fab_lev2.box());

    const CheckDomainBounds lev1_bounds {gm[lev1_idx]};
    const CheckDomainBounds lev2_bounds {gm[lev2_idx]};

    // Extract particle properties
    const auto ptd = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();

    const auto enforceBC = EnforceBC();

    const amrex::Real clight = phys_const.c;
    const amrex::Real inv_clight = 1.0_rt/phys_const.c;
    const amrex::Real inv_clight_SI = 1.0_rt/PhysConstSI::c;
    const amrex::Real inv_c2 = 1.0_rt/(phys_const.c*phys_const.c);
    const amrex::Real charge_mass_ratio = beam.m_charge / beam.m_mass;
    const amrex::Real min_z = gm[0].ProbLo(2) + (slice-gm[0].Domain().smallEnd(2))*gm[0].CellSize(2);
    bool use_external_fields = beam.m_use_external_fields;
    auto external_fields = beam.m_external_fields;

    // Radiation reaction constant
    const amrex::ParticleReal q_over_mc = normalized_units ?
                                  charge_mass_ratio/PhysConstSI::c*PhysConstSI::q_e/PhysConstSI::m_e
                                : charge_mass_ratio/PhysConstSI::c;
    const amrex::ParticleReal RRcoeff = (2.0_rt/3.0_rt)*PhysConstSI::r_e*q_over_mc*q_over_mc;

    // calcuation of E0 in SI units for denormalization
    // using wp_inv to avoid multiplication in kernel
    const amrex::Real wp_inv = normalized_units ? std::sqrt(PhysConstSI::ep0 * PhysConstSI::m_e/
                                     ( static_cast<double>(background_density_SI) *
                                     PhysConstSI::q_e*PhysConstSI::q_e )  ) : 1;
    const amrex::Real E0 = Hipace::m_normalized_units ?
                           PhysConstSI::m_e * PhysConstSI::c / wp_inv / PhysConstSI::q_e : 1;

    // don't include slipped particles in count as they were already pushed
    Hipace::m_num_beam_particles_pushed += double(beam.getNumParticles(WhichBeamSlice::This));

    // Use OMP ParallelFor to use multiple threads when running on CPU
    omp::ParallelFor(
        amrex::TypeList<
            amrex::CompileTimeOptions<0, 1, 2, 3>,
            amrex::CompileTimeOptions<false, true>
        >{}, {
            Hipace::m_depos_order_xy,
            use_external_fields
        },
        beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (int ip, auto depos_order, auto c_use_external_fields) {

            if (!ptd.id(ip).is_valid()) return;

            amrex::Real xp = ptd.pos(0, ip);
            amrex::Real yp = ptd.pos(1, ip);
            amrex::Real zp = ptd.pos(2, ip);
            amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            int i = ptd.idata(BeamIdx::nsubcycles)[ip];

            amrex::RealVect spin {0._rt, 0._rt, 0._rt};
            if (spin_tracking) {
                spin[0] = ptd.m_runtime_rdata[0][ip];
                spin[1] = ptd.m_runtime_rdata[1][ip];
                spin[2] = ptd.m_runtime_rdata[2][ip];
            }

            for (; i < n_subcycles; i++) {

                if (zp < min_z) {
                    // stop pushing particle if it is not on this slice anymore
                    break;
                }

                const amrex::ParticleReal gammap_inv = 1._rt / std::sqrt( 1._rt
                    + (ux*ux + uy*uy + uz*uz)*inv_c2 );

                // first we do half a step in x,y
                // This is not required in z, which is pushed in one step later
                xp += dt * 0.5_rt * ux * gammap_inv;
                yp += dt * 0.5_rt * uy * gammap_inv;

                if (enforceBC(ptd, ip, xp, yp, ux, uy, BeamIdx::w)) return;

                Array3<const amrex::Real> slice_arr = slice_arr_lev0;
                amrex::Real dx_inv = dx_inv_lev0;
                amrex::Real dy_inv = dy_inv_lev0;
                amrex::Real x_pos_offset = x_pos_offset_lev0;
                amrex::Real y_pos_offset = y_pos_offset_lev0;

                if (current_N_level > 2 && lev2_bounds.contains(xp, yp)) {
                    // level 2
                    slice_arr = slice_arr_lev2;
                    dx_inv = dx_inv_lev2;
                    dy_inv = dy_inv_lev2;
                    x_pos_offset = x_pos_offset_lev2;
                    y_pos_offset = y_pos_offset_lev2;
                } else if (current_N_level > 1 && lev1_bounds.contains(xp, yp)) {
                    // level 1
                    slice_arr = slice_arr_lev1;
                    dx_inv = dx_inv_lev1;
                    dy_inv = dy_inv_lev1;
                    x_pos_offset = x_pos_offset_lev1;
                    y_pos_offset = y_pos_offset_lev1;
                }

                // define field at particle position reals
                amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                // field gather for a single particle
                doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                    slice_arr, psi_comp, ez_comp, bx_comp, by_comp, bz_comp,
                    dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                if (c_use_external_fields.value) {
                    ApplyExternalField(xp, yp, zp, time, clight, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                        external_fields);
                }

                // use intermediate fields to calculate next (n+1) transverse momenta
                amrex::ParticleReal ux_next = ux + dt * charge_mass_ratio
                    * ( ExmByp + ( clight - uz * gammap_inv ) * Byp + uy*gammap_inv*Bzp);
                amrex::ParticleReal uy_next = uy + dt * charge_mass_ratio
                    * ( EypBxp + ( uz * gammap_inv - clight ) * Bxp - ux*gammap_inv*Bzp);

                // Now computing new longitudinal momentum
                const amrex::ParticleReal ux_intermediate = ( ux_next + ux ) * 0.5_rt;
                const amrex::ParticleReal uy_intermediate = ( uy_next + uy ) * 0.5_rt;
                const amrex::ParticleReal uz_intermediate = uz
                    + dt * 0.5_rt * charge_mass_ratio * Ezp;

                const amrex::ParticleReal gamma_intermediate_inv = 1._rt / std::sqrt( 1._rt
                    + ( ux_intermediate*ux_intermediate
                       + uy_intermediate*uy_intermediate
                       + uz_intermediate*uz_intermediate )*inv_c2 );

                if (spin_tracking) {
                    const amrex::RealVect E {ExmByp + clight*Byp, EypBxp - clight*Bxp, Ezp};
                    const amrex::RealVect B {Bxp, Byp, Bzp};
                    const amrex::RealVect u {ux_intermediate*inv_clight, uy_intermediate*inv_clight,
                                             uz_intermediate*inv_clight};
                    const amrex::RealVect beta = u*gamma_intermediate_inv;
                    const amrex::Real gamma_inv_p1 =
                        gamma_intermediate_inv / (1._rt + gamma_intermediate_inv);

                    const amrex::RealVect omega = std::abs(charge_mass_ratio) * (
                        B * gamma_intermediate_inv - beta.crossProduct(E) * inv_clight * gamma_inv_p1
                        + spin_anom * (
                            B - gamma_inv_p1 * u * beta.dotProduct(B) - beta.crossProduct(E) * inv_clight
                        )
                    );

                    const amrex::RealVect h = omega * dt * 0.5_rt;
                    const amrex::RealVect s_prime = spin + h.crossProduct(spin);
                    const amrex::Real o = 1._rt / (1._rt + h.dotProduct(h));
                    spin = o * (s_prime + (h.dotProduct(s_prime) * h + h.crossProduct(s_prime)));
                }

                amrex::ParticleReal uz_next = uz + dt * charge_mass_ratio
                    * ( Ezp + ( ux_intermediate * Byp - uy_intermediate * Bxp )
                    * gamma_intermediate_inv );

                if (radiation_reaction) {

                    amrex::ParticleReal Exp = ExmByp + clight*Byp;
                    amrex::ParticleReal Eyp = EypBxp - clight*Bxp;

                    // convert to SI units, no backwards conversion as not used after RR calculation
                    if (normalized_units) {
                        Exp *= E0;
                        Eyp *= E0;
                        Ezp *= E0;
                        Bxp *= E0*inv_clight_SI;
                        Byp *= E0*inv_clight_SI;
                        Bzp *= E0*inv_clight_SI;
                    }
                    const amrex::ParticleReal gamma_intermediate = std::sqrt(
                        1._rt + ( ux_intermediate*ux_intermediate
                                 + uy_intermediate*uy_intermediate
                                 + uz_intermediate*uz_intermediate )*inv_c2 );
                    // Estimation of the velocity at intermediate time
                    const amrex::ParticleReal vx_n = ux_intermediate*gamma_intermediate_inv
                                                     *PhysConstSI::c*inv_clight;
                    const amrex::ParticleReal vy_n = uy_intermediate*gamma_intermediate_inv
                                                     *PhysConstSI::c*inv_clight;
                    const amrex::ParticleReal vz_n = uz_intermediate*gamma_intermediate_inv
                                                     *PhysConstSI::c*inv_clight;
                    // Normalized velocity beta (v/c)
                    const amrex::ParticleReal bx_n = vx_n*inv_clight_SI;
                    const amrex::ParticleReal by_n = vy_n*inv_clight_SI;
                    const amrex::ParticleReal bz_n = vz_n*inv_clight_SI;

                    // Lorentz force over charge
                    const amrex::ParticleReal flx_q = (Exp + vy_n*Bzp - vz_n*Byp);
                    const amrex::ParticleReal fly_q = (Eyp + vz_n*Bxp - vx_n*Bzp);
                    const amrex::ParticleReal flz_q = (Ezp + vx_n*Byp - vy_n*Bxp);
                    const amrex::ParticleReal fl_q2 = flx_q*flx_q + fly_q*fly_q + flz_q*flz_q;

                    // Calculation of auxiliary quantities
                    const amrex::ParticleReal bdotE = (bx_n*Exp + by_n*Eyp + bz_n*Ezp);
                    const amrex::ParticleReal bdotE2 = bdotE*bdotE;
                    const amrex::ParticleReal coeff = gamma_intermediate*gamma_intermediate*(fl_q2-bdotE2);

                    //Compute the components of the RR force
                    const amrex::ParticleReal frx =
                        RRcoeff*(PhysConstSI::c*(fly_q*Bzp - flz_q*Byp) + bdotE*Exp - coeff*bx_n);
                    const amrex::ParticleReal fry =
                        RRcoeff*(PhysConstSI::c*(flz_q*Bxp - flx_q*Bzp) + bdotE*Eyp - coeff*by_n);
                    const amrex::ParticleReal frz =
                        RRcoeff*(PhysConstSI::c*(flx_q*Byp - fly_q*Bxp) + bdotE*Ezp - coeff*bz_n);

                    //Update momentum using the RR force
                    // in normalized units wp_inv normalizes the time step
                    // *clight/inv_clight_SI converts to proper velocity
                    ux_next += frx*dt*wp_inv*clight*inv_clight_SI;
                    uy_next += fry*dt*wp_inv*clight*inv_clight_SI;
                    uz_next += frz*dt*wp_inv*clight*inv_clight_SI;
                }

                /* computing next gamma value */
                const amrex::ParticleReal gamma_next_inv = 1._rt / std::sqrt( 1._rt
                    + ( ux_next*ux_next
                       + uy_next*uy_next
                       + uz_next*uz_next )*inv_c2 );

                /*
                 * computing positions and setting momenta for the next timestep
                 *(n+1)
                 * The longitudinal position is updated here as well, but in
                 * first-order (i.e. without the intermediary half-step) using
                 * a simple Galilean transformation
                 */
                xp += dt * 0.5_rt * ux_next * gamma_next_inv;
                yp += dt * 0.5_rt * uy_next * gamma_next_inv;
                if (do_z_push) zp += dt * ( uz_next * gamma_next_inv - clight );
                ux = ux_next;
                uy = uy_next;
                uz = uz_next;
            } // end for loop over n_subcycles
            if (enforceBC(ptd, ip, xp, yp, ux, uy, BeamIdx::w)) return;
            ptd.pos(0, ip) = xp;
            ptd.pos(1, ip) = yp;
            ptd.pos(2, ip) = zp;
            ptd.idata(BeamIdx::nsubcycles)[ip] = i;
            ptd.rdata(BeamIdx::ux)[ip] = ux;
            ptd.rdata(BeamIdx::uy)[ip] = uy;
            ptd.rdata(BeamIdx::uz)[ip] = uz;

            if (spin_tracking) {
                ptd.m_runtime_rdata[0][ip] = spin[0];
                ptd.m_runtime_rdata[1][ip] = spin[1];
                ptd.m_runtime_rdata[2][ip] = spin[2];
            }
        });
}
