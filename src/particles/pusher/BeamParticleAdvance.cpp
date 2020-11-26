#include "BeamParticleAdvance.H"
#include "particles/ExternalFields.H"
#include "FieldGather.H"
#include "Constants.H"
#include "GetAndSetPosition.H"
#include "HipaceProfilerWrapper.H"

void
AdvanceBeamParticlesSlice (BeamParticleContainer& beam, Fields& fields,
                      amrex::Geometry const& gm, int const lev, const int islice,
                      amrex::DenseBins<BeamParticleContainer::ParticleType>& bins)
{
    HIPACE_PROFILE("AdvanceBeamParticlesSlice()");
    using namespace amrex::literals;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
    const PhysConst phys_const = get_phys_const();

    const amrex::Real dt = Hipace::m_dt;
    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Assumes '2' == 'z' == 'the long dimension'.
        int islice_local = islice - pti.tilebox().smallEnd(2);

        // Extract properties associated with the extent of the current box
        const int depos_order_xy = Hipace::m_depos_order_xy;
        const amrex::Box tilebox = pti.tilebox().grow(
            {depos_order_xy, depos_order_xy,
             Hipace::m_depos_order_z});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields
        const amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
        const amrex::MultiFab exmby(S, amrex::make_alias, FieldComps::ExmBy, 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, FieldComps::EypBx, 1);
        const amrex::MultiFab ez(S, amrex::make_alias, FieldComps::Ez, 1);
        const amrex::MultiFab bx(S, amrex::make_alias, FieldComps::Bx, 1);
        const amrex::MultiFab by(S, amrex::make_alias, FieldComps::By, 1);
        const amrex::MultiFab bz(S, amrex::make_alias, FieldComps::Bz, 1);

        // Extract field array from FabArrays in MultiFabs.
        // (because there is currently no transverse parallelization, the index
        // we want in the slice multifab is always 0. Fix later.
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby[0].array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx[0].array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez[0].array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx[0].array();
        amrex::Array4<const amrex::Real> const& by_arr = by[0].array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz[0].array();

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

        // Extract particle properties
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        const auto uxp = soa.GetRealData(BeamIdx::ux).data();
        const auto uyp = soa.GetRealData(BeamIdx::uy).data();
        const auto uzp = soa.GetRealData(BeamIdx::uz).data();

        const auto getPosition =
            GetParticlePosition<BeamParticleContainer, BeamParticleIterator>(pti);
        const auto setPosition =
            SetParticlePosition<BeamParticleContainer, BeamParticleIterator>(pti);
        const amrex::Real zmin = xyzmin[2];

        // Declare a DenseBins to pass it to doDepositionShapeN, although it will not be used.
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type*
            indices = nullptr;
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const *
            offsets = nullptr;
        indices = bins.permutationPtr();
        offsets = bins.offsetsPtr();
        amrex::DenseBins<BeamParticleContainer::ParticleType>::index_type const
            cell_start = offsets[islice_local], cell_stop = offsets[islice_local+1];
        // The particles that are in slice islice_local are
        // given by the indices[cell_start:cell_stop]

        int const num_particles = cell_stop-cell_start;

        const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
        const amrex::Real charge_mass_ratio = - phys_const.q_e / phys_const.m_e;
        const amrex::Real external_field_strength = Hipace::m_external_field_strength;

        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (long idx) {
                const int ip = indices[cell_start+idx];

                const amrex::ParticleReal gammap = sqrt( 1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                            + uyp[ip]*uyp[ip]*clightsq + uzp[ip]*uzp[ip]*clightsq);

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                // first we do half a step in x,y
                // This is not required in z, which is pushed in one step later
                xp += dt * 0.5_rt * uxp[ip] / gammap;
                yp += dt * 0.5_rt * uyp[ip] / gammap;

                setPosition(ip, xp, yp, zp);

                // define field at particle position reals
                amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                // field gather for a single particle
                doGatherShapeN(xp, yp, zmin,
                               ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                               exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                               dx_arr, xyzmin_arr, lo, depos_order_xy, 0);

                ApplyExternalField(xp, yp, ExmByp, EypBxp, external_field_strength);

                // use intermediate fields to calculate next (n+1) transverse momenta
                const amrex::ParticleReal ux_next = uxp[ip] + dt * charge_mass_ratio
                            * ( ExmByp + ( phys_const.c - uzp[ip] / gammap ) * Byp );
                const amrex::ParticleReal uy_next = uyp[ip] + dt * charge_mass_ratio
                            * ( EypBxp + ( uzp[ip] / gammap - phys_const.c ) * Bxp );

                // Now computing new longitudinal momentum
                const amrex::ParticleReal ux_intermediate = ( ux_next + uxp[ip] ) * 0.5_rt;
                const amrex::ParticleReal uy_intermediate = ( uy_next + uyp[ip] ) * 0.5_rt;
                const amrex::ParticleReal uz_intermediate = uzp[ip]
                                                      + dt * 0.5_rt * charge_mass_ratio * Ezp;

                const amrex::ParticleReal gamma_intermediate = sqrt( 1.0_rt
                                        + ux_intermediate*ux_intermediate*clightsq
                                        + uy_intermediate*uy_intermediate*clightsq
                                        + uz_intermediate*uz_intermediate*clightsq );

                const amrex::ParticleReal uz_next = uzp[ip] + dt * charge_mass_ratio
                          * ( Ezp + ( ux_intermediate * Byp - uy_intermediate * Bxp )
                              / gamma_intermediate );

                /* computing next gamma value */
                const amrex::ParticleReal gamma_next = sqrt( 1.0_rt + uz_next*uz_next*clightsq
                                                 + ux_next*ux_next*clightsq
                                                 + uy_next*uy_next*clightsq );

                /*
                 * computing positions and setting momenta for the next timestep
                 *(n+1)
                 * The longitudinal position is updated here as well, but in
                 * first-order (i.e. without the intermediary half-step) using
                 * a simple Galilean transformation
                 */
                xp += dt * 0.5_rt * ux_next  / gamma_next;
                yp += dt * 0.5_rt * uy_next  / gamma_next;
                zp += dt * ( uz_next  / gamma_next - phys_const.c );
                setPosition(ip, xp, yp, zp);
                uxp[ip] = ux_next;
                uyp[ip] = uy_next;
                uzp[ip] = uz_next;

          }
          );
      }
}


void
CalculateAdaptiveTimeStep (BeamParticleContainer& beam, int const lev)
{
    HIPACE_PROFILE("CalculateAdaptiveTimeStep()");
    using namespace amrex::literals;

    // Extract properties associated with physical size of the box
    const PhysConst phys_const = get_phys_const();

    // const amrex::Real dt = Hipace::m_dt;
    const amrex::Real omega_beta_dyndt = 0.07;
    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {

        // Extract particle properties
        const auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        const auto uzp = soa.GetRealData(BeamIdx::uz).data();
        const auto wp = soa.GetRealData(BeamIdx::w).data();

        amrex::Real min_gamma = 1e100;
        amrex::Gpu::DeviceScalar<amrex::Real> gpu_min_gamma(min_gamma);
        amrex::Real* p_min_gamma = gpu_min_gamma.dataPtr();

        amrex::Real sum_weights = 0;
        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights(sum_weights);
        amrex::Real* p_sum_weights = gpu_sum_weights.dataPtr();

        amrex::Real sum_weights_times_gamma = 0;
        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights_times_gamma(sum_weights_times_gamma);
        amrex::Real* p_sum_weights_times_gamma = gpu_sum_weights_times_gamma.dataPtr();

        amrex::Real sum_weights_times_gamma_squared = 0;
        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights_times_gamma_squared(sum_weights_times_gamma_squared);
        amrex::Real* p_sum_weights_times_gamma_squared = gpu_sum_weights_times_gamma_squared.dataPtr();

        int const num_particles = pti.numParticles();
        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (long ip) {

                amrex::Gpu::Atomic::Add(p_sum_weights, wp[ip]);
                amrex::Gpu::Atomic::Add(p_sum_weights_times_gamma, wp[ip]*uzp[ip]);
                amrex::Gpu::Atomic::Add(p_sum_weights_times_gamma_squared, wp[ip]*uzp[ip]*uzp[ip]);
                amrex::Gpu::Atomic::Min(p_min_gamma, uzp[ip]);

          }
          );
          sum_weights = gpu_sum_weights.dataValue();
          sum_weights_times_gamma = gpu_sum_weights_times_gamma.dataValue();
          sum_weights_times_gamma_squared = gpu_sum_weights_times_gamma_squared.dataValue();
          min_gamma = std::min(min_gamma, gpu_min_gamma.dataValue());

          // if last rank of the pipeline
          // To be fixed for longitudinal parallelization!
          const amrex::Real mean_gamma = sum_weights_times_gamma / sum_weights;
          const amrex::Real sigma_gamma = sqrt(sum_weights_times_gamma_squared / sum_weights - mean_gamma);
          const amrex::Real sigma_gamma_dev = mean_gamma - 4.*sigma_gamma;

          const amrex::Real chosen_min_gamma = std::min( std::max(sigma_gamma_dev, min_gamma), 1e100 );
          if (chosen_min_gamma < 1) amrex::Print()<<"WARNING: beam particles have non-relativistic velocities!";

          if (chosen_min_gamma > 1) // and density above min density
          {
              Hipace::m_dt = sqrt(2.*chosen_min_gamma) * omega_beta_dyndt;
          }

    }
}
