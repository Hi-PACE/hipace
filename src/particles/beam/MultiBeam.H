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
     */
    void DepositCurrentSlice (
        Fields& fields, amrex::Vector<amrex::Geometry> const& geom, const int lev, const int step,
        int islice, const bool do_beam_jx_jy_deposition, const bool do_beam_jz_deposition,
        const bool do_beam_rhomjz_deposition, const int which_slice);

    /** Loop over all beam species and build and return the indices of particles sorted per slice
     * \param[in] ibox box index
     * \param[in] bx 3D box on which per-slice sorting is done
     * \param[in] geom Geometry of the simulation domain
     */
    void
    findParticlesInEachSlice (int ibox, amrex::Box bx, amrex::Geometry const& geom);
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
     */
    void InSituComputeDiags (int step, int islice, int islice_local);
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom);
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

    /** returns the local number of particles of a beam */
    unsigned long long get_local_n_part (int i) const
        {return m_all_beams[i].TotalNumberOfParticles(1,1);}

    /** returns the local number of particles of a beam */
    unsigned long long get_total_num_particles (int i) const
        {return m_all_beams[i].get_total_num_particles();}

    /** \brief Check that all beams have the same number of Real components
     * and return this number */
    int NumRealComps ();

    /** \brief Check that all beams have the same number of Int components
     * and return this number */
    int NumIntComps ();

    /** \brief Store the finest level of every beam particle on which_slice in the cpu() attribute.
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] geom3D Geometry object for the whole domain
     * \param[in] which_slice slice to mark
     * \param[in] islice_local local index of the slice
     */
    void TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                     const int which_slice, const int islice_local);

    /** \brief Store number of particles of each beam in m_n_real_particles */
    void StoreNRealParticles ();

    /** \brief Calculate and return the number of ghost particles in beam ibeam in box it.
     * This is the number of particles in the last slice of it, so ghost particles of it+1.
     *
     * \param[in] ibeam index of the beam
     * \param[in] bx box in which we want to acce particles
     */
    int NGhostParticles (int ibeam, amrex::Box bx);

    /** \brief remove ghost particles, in practice those after the last slice. */
    void RemoveGhosts ();

    void SetIbox (const int ibox);

    /** Return the number of particles in beam ibeam
     *
     * \param[in] ibeam beam index
     */
    int Npart (int ibeam) const {return m_all_beams[ibeam].numParticles();}

    /** \brief copy particles in box it-1 in the ghost buffer at the end of the particle array.
     *
     * The head rank does not receive ghost particles from anyone, but still has to handle them.
     * Besides, slipped particles should also be appended as ghost slices.
     * This function performs both of these tasks.
     *
     * \param[in] it index of the box from which we copy particles to the ghost buffer
     */
    void PackLocalGhostParticles (int it);

    /** \brief getter function for number of real particles (as opposed to ghost particles)
     *
     * \param[in] ibeam index of the beam
     */
    int getNRealParticles (int ibeam) const {return m_n_real_particles[ibeam];}

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
