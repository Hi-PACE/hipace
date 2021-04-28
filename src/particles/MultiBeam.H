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
     * \param[in] geom Geometry object at level lev
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
        Fields& fields, const amrex::Geometry& geom, const int lev, int islice, const amrex::Box bx,
        amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins,
        const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox,
        const bool do_beam_jx_jy_deposition, const int which_slice);

    /** Loop over all beam species and build and return the indices of particles sorted per slice
     * \param[in] lev MR level
     * \param[in] ibox box index
     * \param[in] bx 3D box on which per-slice sorting is done
     * \param[in] geom Geometry of the simulation domain
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     */
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>
    findParticlesInEachSlice (int lev, int ibox, amrex::Box bx, amrex::Geometry& geom,
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
        Fields& fields, amrex::Geometry const& gm, int const lev, const int islice, const amrex::Box bx,
        amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins,
        const amrex::Vector<BoxSorter>& a_box_sorter_vec, const int ibox);

    /** Loop over species and init them
     * \param[in] geom Simulation geometry
     */
    void InitData (const amrex::Geometry& geom);

    /** Return 1 species
     * \param[in] i index of the beam
     */
    BeamParticleContainer& getBeam (int i) {return m_all_beams[i];};

    /** returns the number of beams */
    int get_nbeams () const {return m_nbeams;};

    /** returns the name of a beam */
    std::string get_name (int i) const {return m_all_beams[i].get_name();};

    /** returns the local number of particles of a beam */
    unsigned long long get_local_n_part (int i) const
        {return m_all_beams[i].TotalNumberOfParticles(1,1);};

    /** returns the local number of particles of a beam */
    unsigned long long get_total_num_particles (int i) const
        {return m_all_beams[i].get_total_num_particles();};

    /** \brief Check that all beams have the same number of Real components
     * and return this number */
    int NumRealComps ();

    /** \brief Check that all beams have the same number of Int components
     * and return this number */
    int NumIntComps ();

    void StoreNRealParticles ();
    void RemoveGhosts ();
private:

    amrex::Vector<BeamParticleContainer> m_all_beams; /**< contains all beam containers */
    amrex::Vector<std::string> m_names; /**< names of all beam containers */
    int m_nbeams {0}; /**< number of beam containers */
    /** number of real particles per beam, as opposed to ghost particles */
    amrex::Vector<amrex::Long> m_n_real_particles;
};

#endif // MULTIBEAM_H_
