#include "AdaptiveTimeStep.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

/** \brief describes which double is used for the adaptive time step */
struct WhichDouble {
    enum Comp { Dt=0, MinUz, SumWeights, SumWeightsTimesUz, SumWeightsTimesUzSquared, N };
};

AdaptiveTimeStep::AdaptiveTimeStep ()
{
    amrex::ParmParse ppa("hipace");
    ppa.query("do_adaptive_time_step", m_do_adaptive_time_step);
    ppa.query("nt_per_omega_betatron", m_nt_per_omega_betatron);
}

void
AdaptiveTimeStep::Calculate (amrex::Real& dt, BeamParticleContainer& beam,
                             PlasmaParticleContainer& plasma, int const lev)
{
    HIPACE_PROFILE("CalculateAdaptiveTimeStep()");
    using namespace amrex::literals;

    if (m_do_adaptive_time_step == 0) return;

    // Extract properties associated with physical size of the box
    const PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {

        // Extract particle properties
        const auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        const auto uzp = soa.GetRealData(BeamIdx::uz).data();
        const auto wp = soa.GetRealData(BeamIdx::w).data();

        // if first rank, else receive from upper rank
        m_timestep_data[WhichDouble::SumWeights] = 0.;
        m_timestep_data[WhichDouble::SumWeightsTimesUz] = 0.;
        m_timestep_data[WhichDouble::SumWeightsTimesUzSquared] = 0.;
        m_timestep_data[WhichDouble::MinUz] = 1e100;

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_min_uz(m_timestep_data[WhichDouble::MinUz]);
        amrex::Real* p_min_uz = gpu_min_uz.dataPtr();

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights(
            m_timestep_data[WhichDouble::SumWeights]);
        amrex::Real* p_sum_weights = gpu_sum_weights.dataPtr();

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights_times_uz(
            m_timestep_data[WhichDouble::SumWeightsTimesUz]);
        amrex::Real* p_sum_weights_times_uz = gpu_sum_weights_times_uz.dataPtr();

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights_times_uz_squared(
            m_timestep_data[WhichDouble::SumWeightsTimesUzSquared]);
        amrex::Real* p_sum_weights_times_uz_squared =
            gpu_sum_weights_times_uz_squared.dataPtr();

        int const num_particles = pti.numParticles();
        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (long ip) {

                amrex::Gpu::Atomic::Add(p_sum_weights, wp[ip]);
                amrex::Gpu::Atomic::Add(p_sum_weights_times_uz, wp[ip]*uzp[ip]/phys_const.c);
                amrex::Gpu::Atomic::Add(p_sum_weights_times_uz_squared, wp[ip]*uzp[ip]*uzp[ip]
                                        /phys_const.c/phys_const.c);
                amrex::Gpu::Atomic::Min(p_min_uz, uzp[ip]/phys_const.c);

          }
          );
          m_timestep_data[WhichDouble::SumWeights] = gpu_sum_weights.dataValue();
          m_timestep_data[WhichDouble::SumWeightsTimesUz] = gpu_sum_weights_times_uz.dataValue();
          m_timestep_data[WhichDouble::SumWeightsTimesUzSquared] =
                                               gpu_sum_weights_times_uz_squared.dataValue();
          m_timestep_data[WhichDouble::MinUz] = std::min(m_timestep_data[WhichDouble::MinUz],
                                               gpu_min_uz.dataValue());

          // if last rank of the pipeline
          // To be fixed for longitudinal parallelization!
          const amrex::Real mean_uz = m_timestep_data[WhichDouble::SumWeightsTimesUz]
                                         /m_timestep_data[WhichDouble::SumWeights];
          const amrex::Real sigma_uz = sqrt(m_timestep_data[WhichDouble::SumWeightsTimesUzSquared]
                                          /m_timestep_data[WhichDouble::SumWeights] - mean_uz);
          const amrex::Real sigma_uz_dev = mean_uz - 4.*sigma_uz;
          const amrex::Real chosen_min_uz = std::min( std::max(sigma_uz_dev,
                                               m_timestep_data[WhichDouble::MinUz]), 1e100 );

          if (chosen_min_uz < 1) {
              amrex::Print()<<"WARNING: beam particles have non-relativistic velocities!";
          }

          if (chosen_min_uz > 1) // and density above min density
          {
              const amrex::Real omega_p = sqrt(plasma.m_density * phys_const.q_e*phys_const.q_e
                                          / ( phys_const.ep0*phys_const.m_e ));
              dt = sqrt(2.*chosen_min_uz)/omega_p * m_nt_per_omega_betatron;
          }
    }
}
