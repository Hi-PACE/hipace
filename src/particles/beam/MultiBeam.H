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
#include "particles/sorting/SliceSort.H"
#include "particles/sorting/BoxSort.H"

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
     * \param[in] islice slice index in which the current is stored
     * \param[in] do_beam_jx_jy_deposition whether the beam deposits Jx and Jy
     * \param[in] do_beam_jz_deposition whether the beam deposits Jz
     * \param[in] do_beam_rhomjz_deposition whether the beam deposits rhomjz
     * \param[in] which_slice defines if this or the next slice is handled
     * \param[in] only_highest if the particles deposit only on their highest MR level
     */
    void DepositCurrentSlice (
        Fields& fields, amrex::Vector<amrex::Geometry> const& geom, const int lev, const int step,
        int islice, const bool do_beam_jx_jy_deposition, const bool do_beam_jz_deposition,
        const bool do_beam_rhomjz_deposition, const int which_slice, const bool only_highest=false);

    /** \brief Loop over all beam species and sort particles by box
     *
     * \param[in] a_ba BoxArray object to put the particles into
     * \param[in] a_geom Geometry object with the low corner of the domain
     */
    void
    sortParticlesByBox (const amrex::BoxArray a_ba, const amrex::Geometry& a_geom);

    /** Loop over all beam species and advance slice islice of all beam species
     * \param[in] fields Field object, with 2D slice MultiFabs
     * \param[in] gm Geometry object at level lev
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] islice slice index in which the current is stored
     * \param[in] extEu uniform external electric field applied to beam particles.
                  The components represent Ex-c*By, Ey+c*Bx and Ez respectively.
     * \param[in] extBu Uniform external magnetic field applied to beam particles.
                  The components represent Bx, By and Bz, respectively.
     * \param[in] extEs Slope of a linear external electric field applied to beam particles.
                  The components represent d(Ex-c*By)/dx, d(Ey+c*Bx)/dy and d(Ez)/dz respectively.
                  For the last component, z represents the zeta coordinate zeta = z - c*t.
     * \param[in] extBs Slope of a linear external magnetic field applied to beam particles.
                  The components represent d(Bx)/dy, d(By)/dx and d(Bz)/dz respectively.
                  Note the order of derivatives for the transverse components!
                  For the last component, z represents the zeta coordinate zeta = z - c*t
     */
    void AdvanceBeamParticlesSlice (
        const Fields& fields, amrex::Vector<amrex::Geometry> const& gm, int const current_N_level,
        const int islice, const amrex::RealVect& extEu, const amrex::RealVect& extBu,
        const amrex::RealVect& extEs, const amrex::RealVect& extBs);

    /** Compute reduced beam diagnostics of current slice, store in member variable.
     * \param[in] step time step of simulation
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] islice_local local index of the slice
     * \param[in] max_step maximum time step of simulation
     * \param[in] physical_time physical time at the given step
     * \param[in] max_time maximum time of simulation
     */
    void InSituComputeDiags (int step, int islice, int islice_local,
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
     * \param[in] islice_local local index of the slice
     */
    void TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                     const int which_slice, const int islice_local);

    /** \brief returns if the SALAME algorithm should be used on this slice
     * \param[in] step time step of simulation
     * \param[in] islice slice index of the local box
     */
    bool isSalameNow (const int step, const int islice);

    /** \brief returns if any beam uses the SALAME algorithm
     */
    bool AnySpeciesSalame ();

private:

    amrex::Vector<BeamParticleContainer> m_all_beams; /**< contains all beam containers */
    amrex::Vector<std::string> m_names {"no_beam"}; /**< names of all beam containers */
    int m_nbeams {0}; /**< number of beam containers */
    /** number of real particles per beam, as opposed to ghost particles */
    amrex::Vector<amrex::Long> m_n_real_particles;
};

#endif // MULTIBEAM_H_
