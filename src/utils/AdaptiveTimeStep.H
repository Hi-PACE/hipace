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
    /** uz*mass/charge of the slowest particles */
    amrex::Real m_min_uz_mq = std::numeric_limits<amrex::Real>::max();
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
    /** Number of substeps on which the phase advance is monitored */
    int m_adaptive_phase_substeps = 2000;
    /** Number of beam species for which adaptive time step is computed */
    int m_nbeams = 0;

public:
    /** Whether to use an adaptive time step */
    bool m_do_adaptive_time_step = false;
    /** Whether to Gather Ez to take into account the deceleration of a particle
        when calculating the time step for parallel jobs. */
    bool m_adaptive_gather_ez = false;
    /** Constructor */
    explicit AdaptiveTimeStep (const int nbeams);

    /** Head rank initial time step
     * \param[in,out] dt initial time step
     */
    void BroadcastTimeStep (amrex::Real& dt);

    /** gather min uz from the beam slice
     * \param[in] beams multibeam containing all beams
     * \param[in] initial whether to calculate the initial dt from the beam init slice
     */
    void
    GatherMinUzSlice (MultiBeam& beams, const bool initial);

    /** calculate the adaptive time step based on the beam energy
     * \param[in] t current physical time
     * \param[in,out] dt the time step
     * \param[in] beams multibeam containing all beams
     * \param[in] plasmas multiplasma to get density profile info
     */
    void
    CalculateFromMinUz (amrex::Real t, amrex::Real& dt, MultiBeam& beams, MultiPlasma& plasmas);

    /** gather min Ez from the beam slice
     * \param[in] beams multibeam containing all beams
     * \param[in] geom geometry object
     * \param[in] fields field object
     */
    void
    GatherMinAccSlice (MultiBeam& beams, const amrex::Geometry& geom,
                       const Fields& fields);

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
