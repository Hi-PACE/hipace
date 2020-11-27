#ifndef ADAPTIVETIMESTEP_H_
#define ADAPTIVETIMESTEP_H_

#include "particles/BeamParticleContainer.H"
#include "particles/PlasmaParticleContainer.H"
#include <AMReX_AmrCore.H>

class Hipace;

/** \brief describes which double is used for the adaptive time step */
struct WhichDouble {
    enum Comp { Dt=0, MinUz, SumWeights, SumWeightsTimesUz, SumWeightsTimesUzSquared, N };
};

/** \brief class handling the adaptive time step */
class AdaptiveTimeStep
{
private:

    /** container including dt, min_gamma, sum of weights and the sum of weights times gamma */
    amrex::Real m_container[5] = {0., 1e100, 0., 0., 0.};

    /** Whether to use an adaptive time step */
    bool m_do_adaptive_time_step = false;
    /** Number of time steps per betatron period for the adaptive time step */
    amrex::Real m_nt_per_omega_betatron = 0.07;

public:
    /** Constructor */
    explicit AdaptiveTimeStep ();//(Hipace const* a_hipace);

    /** calculate the adaptive time step based on the beam energy
     * \param[in] beam species on which the time step is based
     * \param[in] plasma plasma container to get the plasma density
     * \param[in] lev MR level
     */
    void
    Calculate (BeamParticleContainer& beam, PlasmaParticleContainer& plasma, int const lev);

};

#endif // ADAPTIVETIMESTEP_H_
