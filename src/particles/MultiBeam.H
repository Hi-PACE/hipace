#ifndef MULTIBEAM_H_
#define MULTIBEAM_H_

#include "BeamParticleContainer.H"
#include "fields/Fields.H"
#include "particles/BinSort.H"

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
     * \param[in] bins Vector (over species) of particles sorted by slices
     */
    void DepositCurrentSlice (
        Fields& fields, const amrex::Geometry& geom, const int lev, int islice,
        amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins);

    /** Loop over all beam species and build and return the indices of particles sorted per slice
     * \param[in] lev MR level
     * \param[in] ibox box index
     * \param[in] bx 3D box on which per-slice sorting is done
     * \param[in] geom Geometry of the simulation domain
     */
    amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>
    findParticlesInEachSlice (int lev, int ibox, amrex::Box bx, amrex::Geometry& geom);

    /** \brief Reorder particles such that those in the same slice are adjacent in memory.
     * \param[in] a_bins Vector<amrex::DenseBins> objects for each beam
     */
    void
    reorderParticlesInEachSlice (
        const amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& a_bins);

    /** Loop over all beam species and advance slice islice of all beam species
     * \param[in] fields Field object, with 2D slice MultiFabs
     * \param[in] gm Geometry object at level lev
     * \param[in] lev MR level
     * \param[in] islice slice index in which the current is stored
     * \param[in] bins Vector (over species) of particles sorted by slices
     */
    void AdvanceBeamParticlesSlice (
        Fields& fields, amrex::Geometry const& gm, int const lev, const int islice,
        amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins);

    /** Loop over species and dump them to plotfile
     * \param[in] filename name of the output file
     */
    void WritePlotFile (const std::string& filename);

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

    /** returns the number of beam particles of the upstream ranks */
    int get_upstream_n_part (int i) const {return m_all_beams[i].get_upstream_n_part();};

#ifdef AMREX_USE_MPI
    /** Loop over beams and pass total number of beam particles on upstream ranks */
    void NotifyNumParticles (MPI_Comm a_comm_z);

    /** Loop over beams and receive total number of beam particles on upstream ranks */
    void WaitNumParticles (MPI_Comm a_comm_z);

    /** Redistribute the particles on a slice, meaning particles, which left the box are returned
     * periodically and if peridicity = 0, their weight is set to 0
     * \param[in] lev MR level
     */
    void RedistributeSlice (int const lev);
#endif

    /** \brief Converts momenta from code convention (gamma * v) to SI (m * gamma * v) and vice-versa. */
    void ConvertUnits (ConvertDirection convert_dir);
private:

    amrex::Vector<BeamParticleContainer> m_all_beams; /**< contains all beam containers */
    amrex::Vector<std::string> m_names; /**< names of all beam containers */
    int m_nbeams; /**< number of beam containers */
};

#endif // MULTIBEAM_H_
