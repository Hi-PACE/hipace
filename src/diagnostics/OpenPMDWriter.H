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
#include <openPMD/openPMD.hpp>

/** \brief class handling the IO with openPMD */
class OpenPMDWriter
{
private:
    /** \brief setup the openPMD parameters do dump the AoS beam data
     *
     * \param[in,out] currSpecies openPMD species to set up
     * \param[in] np total number of particles in the bunch
     */
    void SetupPos(openPMD::ParticleSpecies& currSpecies,
                  const unsigned long long& np);

    /** \brief setup the openPMD parameters do dump the SoA beam data
     *
     * \param[in,out] currSpecies openPMD species to set up
     * \param[in] real_comp_names vector with the names of the real components (weight, ux, uy, uz)
     * \param[in] np total number of particles in the bunch
     */
    void SetupRealProperties(openPMD::ParticleSpecies& currSpecies,
                             const amrex::Vector<std::string>& real_comp_names,
                             const unsigned long long np);

    /** \brief save the SoA beam data to openPMD
     *
     * \param[in,out] pti beam particle iterator of the data to be dumped
     * \param[in,out] currSpecies openPMD species to set up
     * \param[in] offset number of particles which have already been written
     * \param[in] real_comp_names vector with the names of the real components (weight, ux, uy, uz)
     */
    void SaveRealProperty(BeamParticleIterator& pti, openPMD::ParticleSpecies& currSpecies,
                          unsigned long long const offset,
                          amrex::Vector<std::string> const& real_comp_names);

    /** \brief writing openPMD beam particle data
     *
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in,out] iteration openPMD iteration to which the data is written
     */
    void WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration iteration);

    /** \brief writing openPMD field data
     *
     * \param[in] a_fields the field class
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] output_step current iteration to be written to file
     * \param[in] lev MR level
     * \param[in] slice_F_xz whether to use slice IO or not
     * \param[in] varnames list of variable names for the fields (ExmBy, EypBx, Ey, ...)
     * \param[in,out] iteration openPMD iteration to which the data is written
     */
    void WriteFieldData (Fields& a_fields, amrex::Geometry const& geom,
                         const int output_step, const int lev, const bool slice_F_xz,
                         const amrex::Vector< std::string > varnames, openPMD::Iteration iteration);

    /** Named Beam SoA attributes per particle as defined in BeamIdx
     */
    amrex::Vector<std::string> m_real_names {"weighting", "momentum_x", "momentum_y", "momentum_z"};

    /** Parallel openPMD-api Series object for output */
    std::unique_ptr< openPMD::Series > m_outputSeries;
public:
    /** Constructor */
    explicit OpenPMDWriter ();

    /** \brief Initialize diagnostics (collective operation)
     */
    void InitDiagnostics ();

    /** \brief writing openPMD data
     *
     * \param[in] a_fields the field class
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] physical_time Physical time of the currenerationt it.
     * \param[in] output_step current iteration to be written to file
     * \param[in] lev MR level
     * \param[in] slice_F_xz whether to use slice IO or not
     * \param[in] varnames list of variable names for the fields (ExmBy, EypBx, Ey, ...)
     */
    void WriteDiagnostics(Fields& a_fields, MultiBeam& a_multi_beam, amrex::Geometry const& geom,
                          const amrex::Real physical_time, const int output_step, const int lev,
                          const bool slice_F_xz, const amrex::Vector< std::string > varnames);

    void reset () {m_outputSeries.reset();};
};

#endif // HIPACE_USE_OPENPMD

#endif // OPENPMDWRITER_H_
