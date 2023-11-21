/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef MULTIBEAM_H_
#define MULTIBEAM_H_

#include "BeamParticleContainer.H"
#include "fields/Fields.H"

class MultiBeam
{

public:

    /** Constructor
     */
    MultiBeam ();

    /** Destructor
     */
    ~MultiBeam () {}

    /** Loop over all beam species and deposit their current on the 2D XY slice
     * \param[in] fields Field object, with 2D slice MultiFabs
     * \param[in] geom Geometry vector for all levels
     * \param[in] lev MR level
     * \param[in] step time step of simulation
     * \param[in] do_beam_jx_jy_deposition whether the beam deposits Jx and Jy
     * \param[in] do_beam_jz_deposition whether the beam deposits Jz
     * \param[in] do_beam_rhomjz_deposition whether the beam deposits rhomjz
     * \param[in] which_slice defines if this, next or salame slice is handled
     * \param[in] which_beam_slice defines if this or the next beam slice is handled
     * \param[in] only_highest if the particles deposit only on their highest MR level
     */
    void DepositCurrentSlice (
        Fields& fields, amrex::Vector<amrex::Geometry> const& geom, const int lev, const int step,
        const bool do_beam_jx_jy_deposition, const bool do_beam_jz_deposition,
        const bool do_beam_rhomjz_deposition, const int which_slice, const int which_beam_slice,
        const bool only_highest=false);

    void shiftSlippedParticles (const int slice, amrex::Geometry const& geom);

    /** Loop over all beam species and advance slice islice of all beam species
     * \param[in] fields Field object, with 2D slice MultiFabs
     * \param[in] gm Geometry object at level lev
     * \param[in] slice longitudinal slice
     * \param[in] current_N_level number of MR levels active on the current slice
     */
    void AdvanceBeamParticlesSlice (
        const Fields& fields, amrex::Vector<amrex::Geometry> const& gm, const int slice,
        int const current_N_level);

    /** Compute reduced beam diagnostics of current slice, store in member variable.
     * \param[in] step time step of simulation
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] max_step maximum time step of simulation
     * \param[in] physical_time physical time at the given step
     * \param[in] max_time maximum time of simulation
     */
    void InSituComputeDiags (int step, int islice,
                             int max_step, amrex::Real physical_time,
                             amrex::Real max_time);
    /** Write reduced beam diagnostics to file
     * \param[in] step time step of simulation
     * \param[in] time physical time at the given step
     * \param[in] geom Simulation geometry
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom,
                            int max_step, amrex::Real max_time);
    /** Loop over species and init them
     * \param[in] geom Simulation geometry
     * \return physical time at which the simulation will start
     */
    amrex::Real InitData (const amrex::Geometry& geom);

    /** \brief Return 1 species
     * \param[in] i index of the beam
     */
    BeamParticleContainer& getBeam (int i) {return m_all_beams[i];}

    /** \brief Return 1 species
     * \param[in] i index of the beam
     */
    const BeamParticleContainer& getBeam (int i) const {return m_all_beams[i];}

    /** returns the number of beams */
    int get_nbeams () const {return m_nbeams;}

    /** returns the name of a beam */
    std::string get_name (int i) const {return m_all_beams[i].get_name();}

    /** \brief Store the finest level of every beam particle on which_slice in the cpu() attribute.
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] geom3D Geometry object for the whole domain
     * \param[in] which_slice slice to mark
     */
    void TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                     const int which_slice);

    /** \brief Reorder beam particles to speed-up push and current deposition
     * \param[in] beam_slice beam slice to reorder, WhichBeamSlice::This WhichBeamSlice::Next
     * \param[in] step current time step
     * \param[in] slice_geom Geometry object for one xy slice
     */
    void ReorderParticles (int beam_slice, int step, amrex::Geometry& slice_geom);

    /** \brief returns if the SALAME algorithm should be used on this slice
     * \param[in] step time step of simulation
     */
    bool isSalameNow (const int step);

    /** \brief returns if any beam uses the SALAME algorithm
     */
    bool AnySpeciesSalame ();
    amrex::Vector<std::string> m_names {"no_beam"}; /**< names of all beam containers */
    amrex::Vector<BeamParticleContainer> m_all_beams; /**< contains all beam containers */

    void shiftBeamSlices () {
        for (auto& beam : m_all_beams) {
            beam.shiftBeamSlices();
        }
    }

private:

    int m_nbeams {0}; /**< number of beam containers */
    /** number of real particles per beam, as opposed to ghost particles */
    amrex::Vector<amrex::Long> m_n_real_particles;
};

#endif // MULTIBEAM_H_
