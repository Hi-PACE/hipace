#ifndef MULTIBEAM_H_
#define MULTIBEAM_H_

#include "BeamParticleContainer.H"
#include "fields/Fields.H"
#include "particles/BinSort.H"
#include "particles/BoxSort.H"

class MultiBeam
{

public:

    /** Constructor
     * \param[in] amr_core AmrCore object
     */
    MultiBeam (amrex::AmrCore* amr_core);

    /** Destructor
     */
    ~MultiBeam () {}

    /** Loop over all beam species and deposit their current on the 2D XY slice
     * \param[in] fields Field object, with 2D slice MultiFabs
     * \param[in] geom Geometry vector for all levels
     * \param[in] lev MR level
     * \param[in] islice slice index in which the current is stored
     * \param[in] bx current box to calculate in loop over longutidinal boxes
     * \param[in] bins Vector (over species) of particles sorted by slices
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     * \param[in] ibox index of the current box
     * \param[in] do_beam_jx_jy_deposition whether the beam deposits Jx and Jy
     * \param[in] which_slice defines if this or the next slice is handled
     */
    void DepositCurrentSlice (
        Fields& fields, amrex::Vector<amrex::Geometry> const& geom, const int lev, int islice,
        const amrex::Box bx, amrex::Vector<BeamBins> bins,
        const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox,
        const bool do_beam_jx_jy_deposition, const int which_slice);

    /** Loop over all beam species and build and return the indices of particles sorted per slice
     * \param[in] lev MR level
     * \param[in] ibox box index
     * \param[in] bx 3D box on which per-slice sorting is done
     * \param[in] geom Geometry of the simulation domain
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     */
    amrex::Vector<amrex::Vector<BeamBins>>
    findParticlesInEachSlice (int lev, int ibox, amrex::Box bx, const amrex::IntVect& ref_ratio,
                              amrex::Vector<amrex::Geometry> const& geom,
                              const amrex::Vector<BoxSorter>& a_box_sorter_vec);

    /** \brief Loop over all beam species and sort particles by box
     *
     * \param[in] a_box_sorter_vec Vector of BoxSorter objects for each beam species
     * \param[in] a_ba BoxArray object to put the particles into
     * \param[in] a_geom Geometry object with the low corner of the domain
     */
    void
    sortParticlesByBox (
        amrex::Vector<BoxSorter>& a_box_sorter_vec,
        const amrex::BoxArray a_ba, const amrex::Geometry& a_geom);

    /** Loop over all beam species and advance slice islice of all beam species
     * \param[in] fields Field object, with 2D slice MultiFabs
     * \param[in] gm Geometry object at level lev
     * \param[in] lev MR level
     * \param[in] islice slice index in which the current is stored
     * \param[in] bx current box to calculate in loop over longutidinal boxes
     * \param[in] bins Vector (over species) of particles sorted by slices
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     * \param[in] ibox index of the current box
     */
    void AdvanceBeamParticlesSlice (
        Fields& fields, amrex::Geometry const& gm, int const lev, const int islice,
        const amrex::Box bx, amrex::Vector<BeamBins>& bins,
        const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox);

    /** Loop over species and init them
     * \param[in] geom Simulation geometry
     */
    void InitData (const amrex::Geometry& geom);

    /** Return 1 species
     * \param[in] i index of the beam
     */
    BeamParticleContainer& getBeam (int i) {return m_all_beams[i];}

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

    /** \brief Store number of particles of each beam in m_n_real_particles */
    void StoreNRealParticles ();

    /** \brief Calculate and return the number of ghost particles in beam ibeam in box it.
     * This is the number of particles in the last slice of it, so ghost particles of it+1.
     *
     * \param[in] ibeam index of the beam
     * \param[in] bins bins object to access particles per slice
     * \param[in] bx box in which we want to acce particles
     */
    int NGhostParticles (int ibeam, amrex::Vector<BeamBins>& bins, amrex::Box bx);

    /** \brief remove ghost particles, in practice those after the last slice. */
    void RemoveGhosts ();

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
     * \param[in] box_sorters BoxSorter object to access the indices of particles in box it
     */
    void PackLocalGhostParticles (int it, const amrex::Vector<BoxSorter>& box_sorters);

    /** \brief getter function for number of real particles (as opposed to ghost particles)
     *
     * \param[in] ibeam index of the beam
     */
    int getNRealParticles (int ibeam) const {return m_n_real_particles[ibeam];}

    /** \brief Allows beams.all_from_file to specify the input file of all beams that have
     * no injection_type. Also passes down beams.iteration, beams.plasma_density and
     * beams.file_coordinates_xyz to the individual beams if applicable.
     *
     * \param[in] beam_names names of all beams.
     */
    void MultiFromFileMacro (const amrex::Vector<std::string> beam_names);
private:

    amrex::Vector<BeamParticleContainer> m_all_beams; /**< contains all beam containers */
    amrex::Vector<std::string> m_names; /**< names of all beam containers */
    int m_nbeams {0}; /**< number of beam containers */
    /** number of real particles per beam, as opposed to ghost particles */
    amrex::Vector<amrex::Long> m_n_real_particles;
};

#endif // MULTIBEAM_H_
