
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
#include <AMReX_GpuComplex.H>
#include <AMReX_Parser_Y.H>


template <int depos_order>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void InterpolateEzInZ (
    amrex::Real& Ezp,
    const amrex::Real xp, const amrex::Real yp, const amrex::Real zp,
    const amrex::Real x_pos_offset, const amrex::Real y_pos_offset, const amrex::Real min_z,
    const amrex::Real dx_inv, const amrex::Real dy_inv, const amrex::Real dz_inv,
    const int ez_comp_prev, const int ez_comp_next, const Array3<const amrex::Real> slice_arr
)
{
    using namespace amrex::literals;

    // x,y,z direction
    const amrex::Real xmid = (xp-x_pos_offset)*dx_inv;
    const amrex::Real ymid = (yp-y_pos_offset)*dy_inv;
    const amrex::Real zmid = (zp-min_z)*dz_inv-0.5_rt;

    const auto [shape_p, pcell] = shape_factor<2>(zmid, 2);
    const auto [shape_n, ncell] = shape_factor<2>(zmid, 0);

    Ezp *= (1._rt - shape_p - shape_n);

    // Gather Ez field on particle from grid
    for (int iy=0; iy<=depos_order; iy++){
        for (int ix=0; ix<=depos_order; ix++){
            // Compute shape factors
            auto [shape_y, jcell] = shape_factor<depos_order>(ymid, iy);
            auto [shape_x, icell] = shape_factor<depos_order>(xmid, ix);

            Ezp += shape_p * shape_y * shape_x * slice_arr(icell, jcell, ez_comp_prev);
            Ezp += shape_n * shape_y * shape_x * slice_arr(icell, jcell, ez_comp_next);
        }
    }
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void PushSpin (
    amrex::RealVect& spin,
    const amrex::Real ExmByp, const amrex::Real EypBxp, const amrex::Real Ezp,
    const amrex::Real Bxp, const amrex::Real Byp, const amrex::Real Bzp,
    const amrex::Real ux_intermediate, const amrex::Real uy_intermediate,
    const amrex::Real uz_intermediate, const amrex::Real gamma_intermediate_inv,
    const amrex::Real charge_mass_ratio, const amrex::Real dt,
    const amrex::Real spin_anom
)
{
    using namespace amrex::literals;

    const amrex::RealVect E {ExmByp + Byp, EypBxp - Bxp, Ezp};
    const amrex::RealVect B {Bxp, Byp, Bzp};
    const amrex::RealVect u {ux_intermediate, uy_intermediate, uz_intermediate};
    const amrex::RealVect beta = u*gamma_intermediate_inv;
    const amrex::Real gamma_inv_p1 =
        gamma_intermediate_inv / (1._rt + gamma_intermediate_inv);

    const amrex::RealVect omega = std::abs(charge_mass_ratio) * (
        B * gamma_intermediate_inv - beta.crossProduct(E) * gamma_inv_p1
        + spin_anom * (
            B - gamma_inv_p1 * u * beta.dotProduct(B) - beta.crossProduct(E)
        )
    );

    const amrex::RealVect h = omega * dt * 0.5_rt;
    const amrex::RealVect s_prime = spin + h.crossProduct(spin);
    const amrex::Real o = 1._rt / (1._rt + h.dotProduct(h));
    spin = o * (s_prime + (h.dotProduct(s_prime) * h + h.crossProduct(s_prime)));
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ApplyRadiationReaction (
    amrex::Real& ux_next, amrex::Real& uy_next, amrex::Real& uz_next,
    const amrex::Real ExmByp, const amrex::Real EypBxp, const amrex::Real Ezp,
    const amrex::Real Bxp, const amrex::Real Byp, const amrex::Real Bzp,
    const amrex::Real ux_intermediate, const amrex::Real uy_intermediate,
    const amrex::Real uz_intermediate, const amrex::Real gamma_intermediate_inv,
    const amrex::Real dt, const amrex::Real rr_factor
)
{
    using namespace amrex::literals;

    const amrex::Real Exp = ExmByp + Byp;
    const amrex::Real Eyp = EypBxp - Bxp;

    const amrex::Real gamma_intermediate = std::sqrt( 1._rt
        + ux_intermediate*ux_intermediate
        + uy_intermediate*uy_intermediate
        + uz_intermediate*uz_intermediate);

    // Estimation of normalized velocity beta (v/c) at intermediate time
    const amrex::Real bx_n = ux_intermediate * gamma_intermediate_inv;
    const amrex::Real by_n = uy_intermediate * gamma_intermediate_inv;
    const amrex::Real bz_n = uz_intermediate * gamma_intermediate_inv;

    // Lorentz force over charge
    const amrex::Real flx_q = (Exp + by_n*Bzp - bz_n*Byp);
    const amrex::Real fly_q = (Eyp + bz_n*Bxp - bx_n*Bzp);
    const amrex::Real flz_q = (Ezp + bx_n*Byp - by_n*Bxp);
    const amrex::Real fl_q2 = flx_q*flx_q + fly_q*fly_q + flz_q*flz_q;

    // Calculation of auxiliary quantities
    const amrex::Real bdotE = (bx_n*Exp + by_n*Eyp + bz_n*Ezp);
    const amrex::Real bdotE2 = bdotE*bdotE;
    const amrex::Real coeff = gamma_intermediate*gamma_intermediate*(fl_q2-bdotE2);

    // Compute the components of the RR force
    const amrex::Real frx = fly_q*Bzp - flz_q*Byp + bdotE*Exp - coeff*bx_n;
    const amrex::Real fry = flz_q*Bxp - flx_q*Bzp + bdotE*Eyp - coeff*by_n;
    const amrex::Real frz = flx_q*Byp - fly_q*Bxp + bdotE*Ezp - coeff*bz_n;

    // Update momentum using the RR force
    ux_next += dt * rr_factor * frx;
    uy_next += dt * rr_factor * fry;
    uz_next += dt * rr_factor * frz;
}


struct MRLevelData {
    // Array to access fields
    Array3<const amrex::Real> slice_arr;
    // Properties associated with physical size of the box
    amrex::Real dx_inv = 0;
    amrex::Real dy_inv = 0;
    // Offset for converting positions to indexes
    amrex::Real x_pos_offset = 0;
    amrex::Real y_pos_offset = 0;
};


void
AdvanceBeamParticlesSlice (
    BeamParticleContainer& beam, const Fields& fields, amrex::Vector<amrex::Geometry> const& gm,
    const int slice, int const current_N_level, const Helmholtz& helmholtz)
{
    HIPACE_PROFILE("AdvanceBeamParticlesSlice()");
    using namespace amrex::literals;
    const bool use_helmholtz = helmholtz.UseHelmholtz();

    const PhysConst phys_const = get_phys_const();

    const bool do_z_push = beam.m_do_z_push;
    const int n_subcycles = beam.m_n_subcycles;
    const bool radiation_reaction = beam.m_do_radiation_reaction;
    const amrex::Real time = Hipace::GetInstance().m_physical_time;
    const amrex::Real dt = Hipace::GetInstance().m_dt / n_subcycles;
    const bool spin_tracking = beam.m_do_spin_tracking;
    const amrex::Real spin_anom = beam.m_spin_anom;
    amrex::Real* AMREX_RESTRICT undulator_B0 = beam.m_undulator_B0.data();
    amrex::Real* AMREX_RESTRICT undulator_period = beam.m_undulator_period.data();
    amrex::Real* AMREX_RESTRICT undulator_fcK = beam.m_undulator_fcK.data();
    amrex::Real* AMREX_RESTRICT undulator_nperiod = beam.m_undulator_nperiod.data();
    amrex::Real* AMREX_RESTRICT undulator_kx = beam.m_undulator_kx.data();
    amrex::Real* AMREX_RESTRICT undulator_ky = beam.m_undulator_ky.data();
    amrex::Real* AMREX_RESTRICT undulator_z = beam.m_undulator_z.data();
    int nundulator = beam.m_nundulator;
    amrex::Real* AMREX_RESTRICT thinquad_z = beam.m_thinquad_z.data();
    amrex::Real* AMREX_RESTRICT thinquad_K = beam.m_thinquad_K.data();
    int nthinquad = beam.m_nthinquad;
    amrex::Real* AMREX_RESTRICT thickquad_z = beam.m_thickquad_z.data();
    amrex::Real* AMREX_RESTRICT thickquad_l = beam.m_thickquad_l.data();
    amrex::Real* AMREX_RESTRICT thickquad_k1ga = beam.m_thickquad_k1ga.data();
    int nthickquad = beam.m_nthickquad;
    amrex::Real* AMREX_RESTRICT phaseshifter_z = beam.m_phaseshifter_z.data();
    amrex::Real* AMREX_RESTRICT phaseshifter_dz = beam.m_phaseshifter_dz.data();
    int nphaseshifter = beam.m_nphaseshifter;

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
    const amrex::MultiFab& a_mf = helmholtz.getSlices();

    // I suspect we use Ex_n00j00 to push particles but should be Ex_np1j00
    const amrex::GpuArray<int, 3> helm1 {
        WhichHelmholtzSlice::Ex_n00jm1, WhichHelmholtzSlice::Ex_n00j00, WhichHelmholtzSlice::Ex_n00jp1};
    const amrex::GpuArray<int, 3> helm2 {
        WhichHelmholtzSlice::Ex_n00j00, WhichHelmholtzSlice::Ex_n00j00, WhichHelmholtzSlice::Ex_n00j00};
    const amrex::GpuArray<int, 3> helm_comps = helmholtz.InterpZ() ? helm1 : helm2;

    // Imaginary part also needed for Envelope mode
    const amrex::GpuArray<int, 3> helm3 {
        WhichHelmholtzSlice::Ei_n00jm1, WhichHelmholtzSlice::Ei_n00j00, WhichHelmholtzSlice::Ei_n00jp1};
    const amrex::GpuArray<int, 3> helm4 {
        WhichHelmholtzSlice::Ei_n00j00, WhichHelmholtzSlice::Ei_n00j00, WhichHelmholtzSlice::Ei_n00j00};
    const amrex::GpuArray<int, 3> helm_comps_i = helmholtz.InterpZ() ? helm3 : helm4;

    // Beam density, also needed for Envelope mode
    const amrex::GpuArray<int, 3> helm5 {
        WhichHelmholtzSlice::jx_n00jm1, WhichHelmholtzSlice::jx_n00j00, WhichHelmholtzSlice::jx_n00jp1};
    const amrex::GpuArray<int, 3> helm6 {
        WhichHelmholtzSlice::jx_n00j00, WhichHelmholtzSlice::jx_n00j00, WhichHelmholtzSlice::jx_n00j00};
    const amrex::GpuArray<int, 3> helm_comps_d = helmholtz.InterpZ() ? helm5 : helm6;

    // Array3<const amrex::Real> const& a_arr = use_helmholtz ?
    //         a_mf[0].const_array(WhichHelmholtzSlice::Ex_n00j00) : amrex::Array4<const amrex::Real>();
    Array3<const amrex::Real> const& a_arr = use_helmholtz ?
        a_mf[0].const_array() : amrex::Array4<const amrex::Real>();
    const bool helm_mode_is_envelope = helmholtz.ModeIsEnvelope();
    const amrex::Real k = helmholtz.getk0();
    AMREX_ALWAYS_ASSERT(!(use_helmholtz && helm_mode_is_envelope) || nundulator > 0);

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
            amrex::CompileTimeOptions<false, true>,
            amrex::CompileTimeOptions<false, true>
        >{}, {
            Hipace::m_depos_order_xy,
            use_external_fields,
            do_ez_inzerp,
            use_helmholtz
        },
        beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (int ip, auto depos_order, auto c_use_external_fields,
                              auto c_do_ez_inzerp, auto c_use_helmholtz) {

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

            amrex::Real my_time = time + i * dt;
            for (; i < n_subcycles; i++) {

                if (zp < min_z) {
                    // stop pushing particle if it is not on this slice anymore
                    break;
                }

                const amrex::Real gammap_inv = amrex::Math::rsqrt( 1._rt + ux*ux + uy*uy + uz*uz);

                // first we do half a step in x,y
                // This is not required in z, which is pushed in one step later
                amrex::Real xpi = xp;
                amrex::Real ypi = yp;
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
                if (!c_use_helmholtz.value) {
                    doGatherShapeN<depos_order.value>(xp, yp, ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                        slice_arr, psi_comp, ez_comp, bx_comp, by_comp, bz_comp,
                        dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                }

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

                if (c_use_helmholtz.value) {
                    // If Helmholtz unaveraged model, add magnetic force from undulator
                    for (int iu=0; iu<nundulator; iu++) {
                        const amrex::Real zprop = clight*(time+i*dt) + zp/clight*0._rt - undulator_z[iu];
                        const amrex::Real undulator_l = undulator_nperiod[iu]*undulator_period[iu];
                        const amrex::Real ku = 2._rt*MathConst::pi/undulator_period[iu];
                        amrex::Real mag_dz = 0.5_rt*clight*dt;
                        amrex::Real dz_err = clight*dt/10; // 10x smaller than sub-cycled dt
                        // if (zprop + clight*i*dt >= 0 && zprop + clight*i*dt < undulator_l &&
                        if (zprop >= -dz_err && zprop < undulator_l - mag_dz - dz_err && !helm_mode_is_envelope)
                        {
                            amrex::Real mag_B0 = undulator_B0[iu];
                            amrex::Real mag_kx = undulator_kx[iu];
                            amrex::Real mag_ky = undulator_ky[iu];
                            amrex::Real Bx = 0._rt;
                            amrex::Real By = mag_B0*std::cos( ku*zprop + ku*mag_dz );
                            amrex::Real Bz = 0._rt;
                            // Correction for magnetic fields in undulator
                            Bx += mag_B0 * std::cos( ku*zprop + ku*mag_dz ) * mag_kx*mag_kx*xp*yp;
                            By *= (1._rt + 0.5_rt*mag_kx*mag_kx*xp*xp + 0.5_rt*mag_ky*mag_ky*yp*yp);
                            Bz -= mag_B0 * std::sin( ku*zprop + ku*mag_dz ) * ku*yp;
                            Bxp += Bx;
                            Byp += By;
                            Bzp += Bz;
                            ExmByp -= By;
                            EypBxp += Bx;
                        }
                    }
                }

                // use intermediate fields to calculate next (n+1) transverse momenta
                // Main calculation of u{x,y,z}_next and u_{x,y,z}_intermediate starts
                amrex::Real ux_next = ux + dt * charge_mass_ratio
                    * ( ExmByp + ( 1._rt - uz * gammap_inv ) * Byp + uy * gammap_inv * Bzp);
                amrex::Real uy_next = uy + dt * charge_mass_ratio
                    * ( EypBxp - ( 1._rt - uz * gammap_inv ) * Bxp - ux * gammap_inv * Bzp);
                amrex::Real uz_next = uz;

                // Apply kick from thick quadrupole
                for (int iq=0; iq<nthickquad; iq++) {
                    amrex::Real zi = clight*(time+i*dt);
                    amrex::Real zf = clight*(time+i*dt+dt);
                    if (zi <= thickquad_z[iq]+thickquad_l[iq] && zf > thickquad_z[iq] ) {
                        amrex::Real zmax = std::min(thickquad_z[iq]+thickquad_l[iq], zf);
                        amrex::Real zmin = std::max(thickquad_z[iq], zi);
                        ux_next -= thickquad_k1ga[iq] * (zmax-zmin) * xp;
                        uy_next += thickquad_k1ga[iq] * (zmax-zmin) * yp;
                    }
                }

                if (c_use_helmholtz.value) {
                    // If Helmholtz envelope model, push from undulator + Helmholtz field
                    // If Helmholtz unaveraged model, push from Helmholtz
                    amrex::Real betax = ux * gammap_inv;
                    amrex::Real betay = uy * gammap_inv;
                    if (helm_mode_is_envelope) {
                        amrex::Real K = 0._rt;
                        amrex::Real fcK = 0._rt;
                        amrex::Real mag_kx = 0._rt;
                        amrex::Real ku = 2.*MathConst::pi/undulator_period[0];
                        bool in_undulator = false;
                        for (int iu=0; iu<nundulator; iu++) {
                            const amrex::Real zprop = clight*time + zp/clight*0._rt - undulator_z[iu]; // +i*dt? my_time?
                            // Note the index 0 below. Envelope model: the period
                            // is the same for all undulators, the first element in the array.
                            // Likewise for B0 for now. Later, we could let both adjust provided
                            // lr stays constant.
                            const amrex::Real undulator_l = (undulator_nperiod[iu]-1./n_subcycles)*undulator_period[0];
                            if (zprop + clight*i*dt >= 0 && zprop + clight*i*dt < undulator_l)
                            {
                                amrex::Real mag_B0 = undulator_B0[0];
                                amrex::Real mag_period = undulator_period[0];
                                mag_kx = undulator_kx[0];
                                K = phys_const.q_e * mag_B0 * mag_period /
                                    (2*MathConst::pi*phys_const.m_e*phys_const.c);
                                fcK = undulator_fcK[0];
                                in_undulator = true;
                            }
                        }
                        constexpr amrex::GpuComplex<amrex::Real> I(0.,1.);
                        amrex::Real Frp = 0._rt;
                        doHelmholtzGatherShapeN<depos_order.value>(
                            xp, yp, Frp, a_arr, dx_inv, dy_inv,
                            x_pos_offset, y_pos_offset, helm_comps);
                        amrex::Real Fip = 0._rt;
                        doHelmholtzGatherShapeN<depos_order.value>(
                            xp, yp, Fip, a_arr, dx_inv, dy_inv,
                            x_pos_offset, y_pos_offset, helm_comps_i);
                        // ne * q^2 * mu0 / (ga * me) == (omegap_gamma / c)^2
                        amrex::Real nep = 0._rt;
                        doHelmholtzGatherShapeN<depos_order.value>(
                            xp, yp, nep, a_arr, dx_inv, dy_inv,
                            x_pos_offset, y_pos_offset, helm_comps_d);
                        amrex::Real omegap = clight * std::sqrt(nep);
                        amrex::Real omega = std::sqrt( k*k*clight*clight + omegap*omegap );
                        amrex::Real betarsq = betax*betax + betay*betay;
                        // does this impose nsubcycle=1?
                        amrex::Real theta = (k+ku)*zp + ku*clight*my_time;
                        my_time += dt;//  ??
                        // Here we assume gamma_j = gamma from Eq. (2.59) of Reiche's PhD thesis
                        amrex::Real theta_dot =
                            + clight*ku
                            - omega * (1._rt + K*K/2._rt) / 2._rt * gammap_inv * gammap_inv
                            - omega * betarsq / 2._rt
                            - omega * (Frp*Frp+Fip*Fip) / 4._rt * gammap_inv * gammap_inv
                            + omega * fcK * gammap_inv * gammap_inv *
                            ((Frp+I*Fip)*amrex::exp(I*theta)).imag() / 2._rt;
                        // u = p/(mc) = gamma*beta normalized momentum
                        amrex::Real uxdot = - clight * K*K/2._rt * mag_kx*mag_kx * gammap_inv * xp;
                        amrex::Real gammadot =
                            -omega * fcK * gammap_inv * ((Frp+I*Fip)*amrex::exp(I*theta)).real() / 2._rt
                            + 0._rt; // 0 is for longitudinal contribution
                        amrex::Real uzdot = ( 1._rt / gammap_inv * gammadot - ux * uxdot ) / uz;
                        if (in_undulator) {
                            ux_next += dt * uxdot;
                            uz_next += dt * uzdot;
                            if (do_z_push) {
                                amrex::Real betaz = (k + theta_dot/clight) / ( k + ku );
                                zp += dt * clight * (betaz - 1._rt);
                            }
                        } else {
                            if (do_z_push) {
                                amrex::Real betaz = uz * gammap_inv;
                                zp += dt * clight * (betaz - 1._rt);
                            }
                        }
                    } else {
                        amrex::Real betaz = uz * gammap_inv;
                        amrex::Real Frp = 0._rt;
                        doHelmholtzGatherShapeN<depos_order.value>(
                            xp, yp, Frp, a_arr, dx_inv, dy_inv,
                            x_pos_offset, y_pos_offset, helm_comps);
                        Frp *= inv_clight;
                        ux_next += dt * charge_mass_ratio * (1._rt-betaz) * Frp;
                        uz_next += dt * charge_mass_ratio
                            * ( Ezp + ( ux * Byp - uy * Bxp ) * gammap_inv );
                        uz_next += dt * charge_mass_ratio * (   betax   ) * Frp;
                    }
                }

                // Now computing new longitudinal momentum
                const amrex::Real ux_intermediate = ( ux_next + ux ) * 0.5_rt;
                const amrex::Real uy_intermediate = ( uy_next + uy ) * 0.5_rt;
                const amrex::Real uz_intermediate = c_use_helmholtz.value
                    ? ( uz_next + uz ) * 0.5_rt
                    : uz + dt * 0.5_rt * charge_mass_ratio * Ezp;

                const amrex::Real gamma_intermediate_inv = amrex::Math::rsqrt( 1._rt
                    + ux_intermediate*ux_intermediate
                    + uy_intermediate*uy_intermediate
                    + uz_intermediate*uz_intermediate);

                if (!c_use_helmholtz.value) {
                    uz_next += dt * charge_mass_ratio * ( Ezp +
                        ( ux_intermediate*Byp - uy_intermediate*Bxp ) * gamma_intermediate_inv );
                }
                // Main calculation of u{x,y,z}_next and u_{x,y,z}_intermediate ends
                // They may be modified through e.g. radiation reaction below

                if (spin_tracking) {
                    // Update spin
                    PushSpin(spin,
                        ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, ux_intermediate, uy_intermediate,
                        uz_intermediate, gamma_intermediate_inv, charge_mass_ratio, dt, spin_anom);
                }

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
                if (do_z_push && !(c_use_helmholtz.value && helm_mode_is_envelope)) {
                    zp += dt * clight * ( uz_next * gamma_next_inv - 1._rt );
                }

                // Apply thin optics: quadrupoles
                for (int iq=0; iq<nthinquad; iq++) {
                    if (clight*(time+i*dt) <= thinquad_z[iq] && clight*(time+i*dt+dt) > thinquad_z[iq] ) {
                        amrex::Real zi = clight*(time+i*dt);
                        amrex::Real zf = clight*(time+i*dt+dt);
                        amrex::Real dz = clight*dt;
                        const amrex::Real xq = (thinquad_z[iq]-zi)/dz * xp + (zf-thinquad_z[iq])/dz * xpi;
                        const amrex::Real yq = (thinquad_z[iq]-zi)/dz * yp + (zf-thinquad_z[iq])/dz * ypi;
                        ux_next -= thinquad_K[iq] * xq;
                        uy_next += thinquad_K[iq] * yq;
                        xp -= (zf-thinquad_z[iq]) * thinquad_K[iq] * xq  * gamma_next_inv;
                        yp += (zf-thinquad_z[iq]) * thinquad_K[iq] * yq  * gamma_next_inv;
                    }
                }

                // Apply thin optics: phase shifters
                for (int iz=0; iz<nphaseshifter; iz++) {
                    if (clight*(time+i*dt) <= phaseshifter_z[iz] && clight*(time+i*dt+dt) > phaseshifter_z[iz] ) {
                        zp -= phaseshifter_dz[iz];
                    }
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
