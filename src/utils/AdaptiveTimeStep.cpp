#include "AdaptiveTimeStep.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 2000;
}
#endif

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
AdaptiveTimeStep::PassTimeStepInfo (amrex::Real& dt, const int nt, MPI_Comm a_comm_z)
{
    HIPACE_PROFILE("SetTimeStep()");
    // using namespace amrex::literals;

    if (m_do_adaptive_time_step == 0) return;

    const int my_rank_z = amrex::ParallelDescriptor::MyProc();
    const int numprocs_z = amrex::ParallelDescriptor::NProcs();

    if ((nt % numprocs_z == my_rank_z) && (my_rank_z == numprocs_z -1 )) // && (nt > 0) )
    {
            std::cout<<"rank "<< my_rank_z <<" receiving delta t \n"<<std::flush;
        // first rank receives the new dt from last rank
        auto recv_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real));
        MPI_Status status;
        MPI_Recv(recv_buffer, 1,
                 amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 0, comm_z_tag, a_comm_z, &status);
         dt = recv_buffer[WhichDouble::Dt];
         m_timestep_data[WhichDouble::Dt] = recv_buffer[WhichDouble::Dt];
         amrex::The_Pinned_Arena()->free(recv_buffer);
    }

    if ((nt % numprocs_z == 0) && (my_rank_z >= 1))
    {
            // std::cout<<"rank "<< my_rank_z <<" before the send \n"<<std::flush;
        auto send_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
              (sizeof(amrex::Real)*WhichDouble::N);
        send_buffer[WhichDouble::Dt] = dt;
        send_buffer[WhichDouble::SumWeights] = m_timestep_data[WhichDouble::SumWeights];
        send_buffer[WhichDouble::SumWeightsTimesUz] = m_timestep_data[WhichDouble::SumWeightsTimesUz];
        send_buffer[WhichDouble::SumWeightsTimesUzSquared] = m_timestep_data[WhichDouble::SumWeightsTimesUzSquared] ;
        send_buffer[WhichDouble::MinUz] = m_timestep_data[WhichDouble::MinUz];
        // MPI_Status status;
        MPI_Send(send_buffer, WhichDouble::N,
                 amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 my_rank_z-1, comm_z_tag, a_comm_z); //, &status);

                     // std::cout<<"rank "<< my_rank_z <<" after the send \n"<<std::flush;
        amrex::The_Pinned_Arena()->free(send_buffer);
    }


}

void
AdaptiveTimeStep::Calculate (amrex::Real& dt, const int nt, BeamParticleContainer& beam,
                             PlasmaParticleContainer& plasma, int const lev, MPI_Comm a_comm_z)
{
    HIPACE_PROFILE("CalculateAdaptiveTimeStep()");
    using namespace amrex::literals;

    if (m_do_adaptive_time_step == 0) return;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( plasma.m_density != 0,
        "A plasma density must be specified to use an adaptive time step.");

    const int my_rank_z = amrex::ParallelDescriptor::MyProc();
    const int numprocs_z = amrex::ParallelDescriptor::NProcs();

    if (nt % numprocs_z != 0) return;
    // Extract properties associated with physical size of the box
    const PhysConst phys_const = get_phys_const();

    if (my_rank_z == numprocs_z-1) {
        m_timestep_data[WhichDouble::SumWeights] = 0.;
        m_timestep_data[WhichDouble::SumWeightsTimesUz] = 0.;
        m_timestep_data[WhichDouble::SumWeightsTimesUzSquared] = 0.;
        m_timestep_data[WhichDouble::MinUz] = 1e100;
    } else {
        auto recv_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real)*WhichDouble::N);
        MPI_Status status;
        MPI_Recv(recv_buffer, WhichDouble::N,
                 amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 my_rank_z+1, comm_z_tag, a_comm_z, &status);

         m_timestep_data[WhichDouble::SumWeights] = recv_buffer[WhichDouble::SumWeights];
         m_timestep_data[WhichDouble::SumWeightsTimesUz] = recv_buffer[WhichDouble::SumWeightsTimesUz];
         m_timestep_data[WhichDouble::SumWeightsTimesUzSquared] = recv_buffer[WhichDouble::SumWeightsTimesUzSquared];
         m_timestep_data[WhichDouble::MinUz] = recv_buffer[WhichDouble::MinUz];
         dt = recv_buffer[WhichDouble::Dt];

         std::cout<<"Rank " << my_rank_z << " received new timestep " << dt << "\n";
         amrex::The_Pinned_Arena()->free(recv_buffer);
    }

    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {

        // Extract particle properties
        const auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        const auto uzp = soa.GetRealData(BeamIdx::uz).data();
        const auto wp = soa.GetRealData(BeamIdx::w).data();

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

    }
    // if last rank of the pipeline
  if (my_rank_z == 0 )
  {
      const amrex::Real mean_uz = m_timestep_data[WhichDouble::SumWeightsTimesUz]
                                       /m_timestep_data[WhichDouble::SumWeights];
      const amrex::Real sigma_uz = sqrt(m_timestep_data[WhichDouble::SumWeightsTimesUzSquared]
                                        /m_timestep_data[WhichDouble::SumWeights] - mean_uz);
      const amrex::Real sigma_uz_dev = mean_uz - 4.*sigma_uz;
      const amrex::Real chosen_min_uz = std::min( std::max(sigma_uz_dev,
                                             m_timestep_data[WhichDouble::MinUz]), 1e100 );

      std::cout << "min gamma " << chosen_min_uz << "\n";
      if (chosen_min_uz < 1) {
            amrex::Print()<<"WARNING: beam particles have non-relativistic velocities!";
      }

      amrex::Real new_dt = dt;
      if (chosen_min_uz > 1) // and density above min density
      {
          const amrex::Real omega_p = sqrt(plasma.m_density * phys_const.q_e*phys_const.q_e
                                        / ( phys_const.ep0*phys_const.m_e ));
          new_dt = sqrt(2.*chosen_min_uz)/omega_p * m_nt_per_omega_betatron;
      }

        // send to first rank
      auto send_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real));
      send_buffer[WhichDouble::Dt] = new_dt;
      // MPI_Status status;
      MPI_Send(send_buffer, 1,
               amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
               numprocs_z-1, comm_z_tag, a_comm_z); //, &status);
      amrex::The_Pinned_Arena()->free(send_buffer);
  }
}
