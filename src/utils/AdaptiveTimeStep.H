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
    amrex::Real m_nt_per_betatron = 20.;
    /** Upper bound of the time step. Avoid gigantic time step(s) when beam starts near vacuum */
    amrex::Real m_dt_max = std::numeric_limits<amrex::Real>::infinity();
    /** uz of the slowest particles */
    amrex::Real m_min_uz = std::numeric_limits<amrex::Real>::max();
    /** Threshold beam momentum, below which the time step is not decreased */
    amrex::Real m_threshold_uz = 2.;
    /** Whether to predict the next time steps. More accurate for parallel simulations */
    bool m_adaptive_predict_step = true;
    /** If true, a test on the phase advance sets the time step so it matches the phase advance
     * expected for a uniform plasma. Relevant in the presence of density gradients.
     * The tolerance on the phase advance difference is controlled by m_adaptive_phase_tolerance */
    bool m_adaptive_control_phase_advance = true;
    /** Phase shift tolerance. Relevant when density gradients are present.
     * Lower is more accurate. */
    amrex::Real m_adaptive_phase_tolerance = 4.e-4;
    /** Number of substeps on which the phase advance is monitored. */
    int m_adaptive_phase_substeps = 2000;

public:
    /** Whether to use an adaptive time step */
    bool m_do_adaptive_time_step = false;
    /** Constructor */
    explicit AdaptiveTimeStep (const int nbeams);

#ifdef AMREX_USE_MPI
    /** Head rank initial time step
     * \param[in,out] dt initial time step
     * \param[in] a_comm_z longitudinal MPI communicator
     * \param[in] a_numprocs_z number of ranks in the z direction
     */
    void BroadcastTimeStep (amrex::Real& dt, MPI_Comm a_comm_z, int a_numprocs_z);
#endif

    /** calculate the adaptive time step based on the beam energy
     * \param[in] t current physical time
     * \param[in,out] dt the time step
     * \param[in] beams multibeam containing all beams
     * \param[in] plasmas multiplasma to get density profile info
     * \param[in] geom geometry object
     * \param[in] fields field object
     * \param[in] it current box number
     * \param[in] initial whether to calculate the initial dt
     */
    void
    Calculate (amrex::Real t, amrex::Real& dt, MultiBeam& beams, MultiPlasma& plasmas,
               const amrex::Geometry& geom, const Fields& fields, const int it=0,
               const bool initial=true);

    /** calculate the adaptive time step based on the beam energy
     * \param[in] beams multibeam containing all beams
     * \param[in] geom geometry object
     * \param[in] fields field object
     * \param[in] islice current slice number
     */
    void
    GatherMinAccSlice (MultiBeam& beams, const amrex::Geometry& geom,
                       const Fields& fields, const int islice);

    /** Right before starting a time step, correct its dt to account for local plasma density
     * and resolve density gradients.
     * \param[in] t current physical time
     * \param[in,out] dt the time step
     * \param[in] plasmas multiplasma to get density profile info
     */
    void
    CalculateFromDensity (amrex::Real t, amrex::Real& dt, MultiPlasma& plasmas);
};

#endif // ADAPTIVETIMESTEP_H_
