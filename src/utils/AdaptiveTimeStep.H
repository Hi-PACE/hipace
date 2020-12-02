#ifndef ADAPTIVETIMESTEP_H_
#define ADAPTIVETIMESTEP_H_

#include "particles/BeamParticleContainer.H"
#include "particles/PlasmaParticleContainer.H"
#include <AMReX_AmrCore.H>

/** \brief class handling the adaptive time step */
class AdaptiveTimeStep
{
private:

    /** container including dt, min_gamma, sum of weights and the sum of weights times gamma */
    amrex::Real m_timestep_data[5] = {0., 1e30, 0., 0., 0.};

    /** Whether to use an adaptive time step */
    bool m_do_adaptive_time_step = false;
    /** Number of time steps per betatron period for the adaptive time step */
    amrex::Real m_nt_per_omega_betatron = 0.07;

public:
    /** Constructor */
    explicit AdaptiveTimeStep ();

    /** calculate the adaptive time step based on the beam energy
     * \param[in,out] dt the time step
     * \param[in] nt number of the current time step
     * \param[in] beam species on which the time step is based
     * \param[in] plasma plasma container to get the plasma density
     * \param[in] lev MR level
     * \param[in] a_comm_z longitudinal MPI communicator
     */
    void
    Calculate (amrex::Real& dt, const int nt, BeamParticleContainer& beam,
                                PlasmaParticleContainer& plasma, int const lev, MPI_Comm a_comm_z);

    /** pass the time step data to the next lower proc in the pipeline
     * \param[in] nt number of the current time step
     * \param[in] a_comm_z longitudinal MPI communicator
     */
    void
    PassTimeStepInfo (const int nt, MPI_Comm a_comm_z);

};

#endif // ADAPTIVETIMESTEP_H_
