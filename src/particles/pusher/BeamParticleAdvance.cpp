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

void
AdvanceBeamParticlesSlice (BeamParticleContainer& beam, const Fields& fields,
                           amrex::Vector<amrex::Geometry> const& gm, int const current_N_level,
                           const int islice_local, const amrex::RealVect& extEu,const amrex::RealVect& extBu,const amrex::RealVect& extEs,const amrex::RealVect& extBs)
{
    HIPACE_PROFILE("AdvanceBeamParticlesSlice()");
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    const bool do_z_push = beam.m_do_z_push;
    const int n_subcycles = beam.m_n_subcycles;
    const amrex::Real dt = Hipace::m_dt / n_subcycles;

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

    const amrex::Real lo_x_lev1 = gm[lev1_idx].ProbLo(0);
    const amrex::Real lo_x_lev2 = gm[lev2_idx].ProbLo(0);

    const amrex::Real hi_x_lev1 = gm[lev1_idx].ProbHi(0);
    const amrex::Real hi_x_lev2 = gm[lev2_idx].ProbHi(0);

    const amrex::Real lo_y_lev1 = gm[lev1_idx].ProbLo(1);
    const amrex::Real lo_y_lev2 = gm[lev2_idx].ProbLo(1);

    const amrex::Real hi_y_lev1 = gm[lev1_idx].ProbHi(1);
    const amrex::Real hi_y_lev2 = gm[lev2_idx].ProbHi(1);

    // Extract particle properties
    const int offset = beam.m_box_sorter.boxOffsetsPtr()[beam.m_ibox];
    const auto ptd = beam.getParticleTileData();

    const auto setPositionEnforceBC = EnforceBCandSetPos<BeamParticleContainer>(gm[0]);

    // Declare a DenseBins to pass it to doDepositionShapeN, although it will not be used.
    BeamBins::index_type const * const indices = beam.m_slice_bins.permutationPtr();
    BeamBins::index_type const * const offsets = beam.m_slice_bins.offsetsPtrCpu();
    BeamBins::index_type const
        cell_start = offsets[islice_local], cell_stop = offsets[islice_local+1];
    // The particles that are in slice islice_local are
    // given by the indices[cell_start:cell_stop]

    int const num_particles = cell_stop-cell_start;

    const amrex::Real clight = phys_const.c;
    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
    const amrex::Real charge_mass_ratio = beam.m_charge / beam.m_mass;
    const amrex::RealVect external_E_uniform = extEu;
    const amrex::RealVect external_B_uniform = extBu;
    const amrex::RealVect external_E_slope = extEs;
    const amrex::RealVect external_B_slope = extBs;

    amrex::ParallelFor(
        amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
        {Hipace::m_depos_order_xy},
        num_particles,
        [=] AMREX_GPU_DEVICE (long idx, auto depos_order) {
            const int ip = indices[cell_start+idx] + offset;

            if (ptd.id(ip) < 0) return;

            amrex::Real xp = ptd.pos(0, ip);
            amrex::Real yp = ptd.pos(1, ip);
            amrex::Real zp = ptd.pos(2, ip);
            amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            for (int i = 0; i < n_subcycles; i++) {

                const amrex::ParticleReal gammap_inv = 1._rt / std::sqrt( 1._rt
                    + ux*ux*clightsq
                    + uy*uy*clightsq
                    + uz*uz*clightsq );

                // first we do half a step in x,y
                // This is not required in z, which is pushed in one step later
                xp += dt * 0.5_rt * ux * gammap_inv;
                yp += dt * 0.5_rt * uy * gammap_inv;

                if (setPositionEnforceBC(ptd, ip, xp, yp, zp)) return;

                Array3<const amrex::Real> slice_arr = slice_arr_lev0;
                amrex::Real dx_inv = dx_inv_lev0;
                amrex::Real dy_inv = dy_inv_lev0;
                amrex::Real x_pos_offset = x_pos_offset_lev0;
                amrex::Real y_pos_offset = y_pos_offset_lev0;

                if (current_N_level > 2 &&
                    lo_x_lev2 < xp && xp < hi_x_lev2 &&
                    lo_y_lev2 < yp && yp < hi_y_lev2) {
                    // level 2
                    slice_arr = slice_arr_lev2;
                    dx_inv = dx_inv_lev2;
                    dy_inv = dy_inv_lev2;
                    x_pos_offset = x_pos_offset_lev2;
                    y_pos_offset = y_pos_offset_lev2;
                } else if (current_N_level > 1 &&
                    lo_x_lev1 < xp && xp < hi_x_lev1 &&
                    lo_y_lev1 < yp && yp < hi_y_lev1) {
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

                ApplyExternalField(xp, yp, zp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                    external_E_uniform, external_B_uniform, external_E_slope, external_B_slope);

                // use intermediate fields to calculate next (n+1) transverse momenta
                const amrex::ParticleReal ux_next = ux + dt * charge_mass_ratio
                    * ( ExmByp + ( clight - uz * gammap_inv ) * Byp );
                const amrex::ParticleReal uy_next = uy + dt * charge_mass_ratio
                    * ( EypBxp + ( uz * gammap_inv - clight ) * Bxp );

                // Now computing new longitudinal momentum
                const amrex::ParticleReal ux_intermediate = ( ux_next + ux ) * 0.5_rt;
                const amrex::ParticleReal uy_intermediate = ( uy_next + uy ) * 0.5_rt;
                const amrex::ParticleReal uz_intermediate = uz
                    + dt * 0.5_rt * charge_mass_ratio * Ezp;

                const amrex::ParticleReal gamma_intermediate_inv = 1._rt / std::sqrt( 1._rt
                    + ux_intermediate*ux_intermediate*clightsq
                    + uy_intermediate*uy_intermediate*clightsq
                    + uz_intermediate*uz_intermediate*clightsq );

                const amrex::ParticleReal uz_next = uz + dt * charge_mass_ratio
                    * ( Ezp + ( ux_intermediate * Byp - uy_intermediate * Bxp )
                    * gamma_intermediate_inv );

                /* computing next gamma value */
                const amrex::ParticleReal gamma_next_inv = 1._rt / std::sqrt( 1._rt
                    + ux_next*ux_next*clightsq
                    + uy_next*uy_next*clightsq
                    + uz_next*uz_next*clightsq );

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
                if (setPositionEnforceBC(ptd, ip, xp, yp, zp)) return;
                ux = ux_next;
                uy = uy_next;
                uz = uz_next;
            } // end for loop over n_subcycles
            ptd.rdata(BeamIdx::ux)[ip] = ux;
            ptd.rdata(BeamIdx::uy)[ip] = uy;
            ptd.rdata(BeamIdx::uz)[ip] = uz;
        });
}
