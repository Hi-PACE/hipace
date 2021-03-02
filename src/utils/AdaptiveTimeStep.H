#ifndef ADAPTIVETIMESTEP_H_
#define ADAPTIVETIMESTEP_H_

#include "particles/MultiBeam.H"
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
     * \param[in] beams multibeam containing all beams
     * \param[in] plasma plasma container to get the plasma density
     * \param[in] it current box number
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     */
    void
    Calculate (amrex::Real& dt, MultiBeam& beams, PlasmaParticleContainer& plasma, const int it,
               const amrex::Vector<BoxSorter>& a_box_sorter_vec);

};

#endif // ADAPTIVETIMESTEP_H_
