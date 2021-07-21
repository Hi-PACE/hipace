#ifndef OPENPMDWRITER_H_
#define OPENPMDWRITER_H_

#include "particles/MultiBeam.H"
#include "particles/BeamParticleContainer.H"

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_AmrCore.H>

#include <cstdint>
#include <vector>

#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

/** \brief Whether the beam, the field data is written, or if it is just flushing the stored data */
enum struct OpenPMDWriterCallType { beams, fields };

#ifdef HIPACE_USE_OPENPMD
/** \brief class handling the IO with openPMD */
class OpenPMDWriter
{
private:
    /** \brief setup the openPMD parameters do dump the AoS beam data
     *
     * \param[in,out] currSpecies openPMD species to set up
     * \param[in] beam beam particle container to get mass and charge
     * \param[in] np total number of particles in the bunch
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     */
    void SetupPos(openPMD::ParticleSpecies& currSpecies, BeamParticleContainer& beam,
                  const unsigned long long& np, const amrex::Geometry& geom);

    /** \brief setup the openPMD parameters do dump the SoA beam data
     *
     * \param[in,out] currSpecies openPMD species to set up
     * \param[in] real_comp_names vector with the names of the real components (weight, ux, uy, uz)
     * \param[in] np total number of particles in the bunch
     */
    void SetupRealProperties (openPMD::ParticleSpecies& currSpecies,
                              const amrex::Vector<std::string>& real_comp_names,
                              const unsigned long long np);

    /** \brief save the SoA beam data to openPMD
     *
     * \param[in,out] pti beam particle iterator of the data to be dumped
     * \param[in,out] currSpecies openPMD species to set up
     * \param[in] offset number of particles which have already been written
     * \param[in] real_comp_names vector with the names of the real components (weight, ux, uy, uz)
     * \param[in] box_offset offset for the particles in the current box
     * \param[in] numParticleOnTile number of particles in this box
     */
    void SaveRealProperty (BeamParticleContainer& pc, openPMD::ParticleSpecies& currSpecies,
                           unsigned long long const offset,
                           amrex::Vector<std::string> const& real_comp_names,
                           unsigned long long const box_offset,
                           const unsigned long long numParticleOnTile);

    /** \brief writing openPMD beam particle data
     *
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in,out] iteration openPMD iteration to which the data is written
     * \param[in] output_step current time step to dump
     * \param[in] it current box number
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] beamnames list of the names of the beam to be written to file
     * \param[in] lev MR level
     */
    void WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration iteration,
                                const int output_step, const int it,
                                const amrex::Vector<BoxSorter>& a_box_sorter_vec,
                                const amrex::Geometry& geom,
                                const amrex::Vector< std::string > beamnames, const int lev);

    /** \brief writing openPMD field data
     *
     * \param[in] fab the FArrayBox to dump
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] slice_dir direction of slicing. 0=x, 1=y, -1=no slicing (3D array)
     * \param[in] varnames list of variable names for the fields (ExmBy, EypBx, Ey, ...)
     * \param[in,out] iteration openPMD iteration to which the data is written
     * \param[in] output_step current time step to dump
     * \param[in] lev MR level
     */
    void WriteFieldData (amrex::FArrayBox const& fab, amrex::Vector<amrex::Geometry> const& geom,
                         const int slice_dir, const amrex::Vector< std::string > varnames,
                         openPMD::Iteration iteration, const int output_step, const int lev);

    /** Named Beam SoA attributes per particle as defined in BeamIdx
     */
    amrex::Vector<std::string> m_real_names {"weighting", "momentum_x", "momentum_y", "momentum_z"};

    /** vector over levels of openPMD-api Series object for output */
    amrex::Vector<std::unique_ptr< openPMD::Series >> m_outputSeries;

    /** openPMD backend: h5, bp, or json. Default depends on what is available */
    std::string m_openpmd_backend = "default";

    /** Last iteration that was written to file.
     * This is stored to make sure we don't write the last iteration multiple times. */
    amrex::Vector<int> m_last_output_dumped;

    /** vector of length nbeams with the numbers of particles already written to file */
    amrex::Vector<uint64_t> m_offset;
    /** vector of length nbeams with the temporary numbers of particles already written to file */
    amrex::Vector<uint64_t> m_tmp_offset;
public:
    /** Constructor */
    explicit OpenPMDWriter ();

    /** \brief Initialize diagnostics (collective operation)
     *
     * \param[in] output_step current iteration
     * \param[in] output_period output period
     * \param[in] max_step maximum time step of the simulation
     * \param[in] nlev number of mesh refinement levels
     */
    void InitDiagnostics (const int output_step, const int output_period, const int max_step,
                          const int nlev);

    /** \brief writing openPMD data
     *
     * \param[in] a_mf Vector (levels) of FArrayBoxes
     * \param[in] a_multi_beams multi beam container which is written to openPMD file
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] physical_time Physical time of the currenerationt it.
     * \param[in] output_step current iteration to be written to file
     * \param[in] nlev number of MR levels
     * \param[in] slice_dir direction of slicing. 0=x, 1=y, -1=no slicing (3D array)
     * \param[in] varnames list of variable names for the fields (ExmBy, EypBx, Ey, ...)
     * \param[in] beamnames list of the names of the beam to be written to file
     * \param[in] it current box number
     * \param[in] a_box_sorter_vec Vector (over species) of particles sorted by box
     * \param[in] geom3D 3D Geometry of the simulation, to get the cell size etc.
     * \param[in] call_type whether the beams or the fields should be written to file
     */
    void WriteDiagnostics(
        amrex::Vector<amrex::FArrayBox> const& a_mf, MultiBeam& a_multi_beam,
        amrex::Vector<amrex::Geometry> const& geom,
        const amrex::Real physical_time, const int output_step, const int nlev,
        const int slice_dir, const amrex::Vector< std::string > varnames,
        const amrex::Vector< std::string > beamnames, const int it,
        const amrex::Vector<BoxSorter>& a_box_sorter_vec, amrex::Vector<amrex::Geometry> const& geom3D,
        const OpenPMDWriterCallType call_type);

    /** \brief Resets the openPMD series of all levels */
    void reset ();

    /** Prefix/path for the output files */
    std::string m_file_prefix = "diags/hdf5";

    /** Temporary workaround to display normalized momentum correctly */
    bool m_openpmd_viewer_workaround = true;
};

#endif // HIPACE_USE_OPENPMD

#endif // OPENPMDWRITER_H_
