/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef ADAPTIVETIMESTEP_H_
#define ADAPTIVETIMESTEP_H_

#include "particles/beam/MultiBeam.H"
#include "particles/plasma/MultiPlasma.H"
#include "particles/beam/BeamParticleContainer.H"
#include <AMReX_AmrCore.H>

/** \brief class handling the adaptive time step */
class AdaptiveTimeStep
{
private:

    /** container including dt, min_gamma, sum of weights and the sum of weights times gamma */
    amrex::Vector<amrex::Vector<amrex::Real>> m_timestep_data;

    /** Number of time steps per betatron period for the adaptive time step */
    amrex::Real m_nt_per_betatron = 40.;
    amrex::Real m_dt_max = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_min_uz = std::numeric_limits<amrex::Real>::max();
    amrex::Real m_adaptive_phase_tolerance = 0.03;

public:
    /** Whether to use an adaptive time step */
    bool m_do_adaptive_time_step = false;
    /** Constructor */
    explicit AdaptiveTimeStep (const int nbeams);

#ifdef AMREX_USE_MPI
    /** sends the calculated initial time step to the rank downstream
     * \param[in] dt initial time step
     * \param[in] a_comm_z longitudinal MPI communicator
     */
    void NotifyTimeStep (amrex::Real dt, MPI_Comm a_comm_z);

    /** receives the calculated initial time step to thefrom the rank upstream
     * \param[in,out] dt initial time step
     * \param[in] a_comm_z longitudinal MPI communicator
     */
    void WaitTimeStep (amrex::Real& dt, MPI_Comm a_comm_z);
#endif

    /** calculate the adaptive time step based on the beam energy
     * \param[in,out] dt the time step
     * \param[in] beams multibeam containing all beams
     * \param[in] plasma_density maximum plasma density
     * \param[in] it current box number
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     * \param[in] initial whether to calculate the initial dt
     */
    void
    Calculate (amrex::Real& dt, MultiBeam& beams, amrex::Real plasma_density, const int it=0,
               const amrex::Vector<BoxSorter>& a_box_sorter_vec={}, const bool initial=true);

    void
    CalculateFromDensity (amrex::Real t, amrex::Real& dt, MultiPlasma& plasmas);
};

#endif // ADAPTIVETIMESTEP_H_
