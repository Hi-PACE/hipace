/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef OPENPMDWRITER_H_
#define OPENPMDWRITER_H_

#include "diagnostics/Diagnostic.H"
#include "particles/beam/MultiBeam.H"
#include "particles/beam/BeamParticleContainer.H"

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

    /** \brief writing openPMD beam particle data
     *
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in,out] iteration openPMD iteration to which the data is written
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] beamnames list of the names of the beam to be written to file
     */
    void WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration iteration,
                                const amrex::Geometry& geom,
                                const amrex::Vector< std::string > beamnames);

    /** \brief writing openPMD field data
     *
     * \param[in] fab the FArrayBox to dump
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] slice_dir direction of slicing. 0=x, 1=y, -1=no slicing (3D array)
     * \param[in] varnames list of variable names for the fields (ExmBy, EypBx, Ey, ...)
     * \param[in,out] iteration openPMD iteration to which the data is written
     */
    void WriteFieldData (amrex::FArrayBox const& fab, amrex::Geometry const& geom,
                         const int slice_dir, const amrex::Vector< std::string > varnames,
                         openPMD::Iteration iteration);

    /** Named Beam SoA attributes per particle as defined in BeamIdx
     */
    amrex::Vector<std::string> m_int_names {
        "id"
    };

    /** Named Beam SoA attributes per particle as defined in BeamIdx
     */
    amrex::Vector<std::string> m_real_names {
        "position_x", "position_y", "position_z",
        "weighting",
        "momentum_x", "momentum_y", "momentum_z"
    };

    /** vector over levels of openPMD-api Series object for output */
    std::unique_ptr< openPMD::Series > m_outputSeries;

    /** openPMD backend: h5, bp, or json. Default depends on what is available */
    std::string m_openpmd_backend = "default";

    /** vector of length nbeams with the numbers of particles already written to file */
    amrex::Vector<uint64_t> m_offset;

    std::vector<std::vector<std::shared_ptr<int>>> m_int_beam_data {};
    std::vector<std::vector<std::shared_ptr<uint64_t>>> m_uint64_beam_data {};
    std::vector<std::vector<std::shared_ptr<amrex::ParticleReal>>> m_real_beam_data {};

public:
    /** Constructor */
    explicit OpenPMDWriter ();

    /** \brief Initialize diagnostics (collective operation)
     */
    void InitDiagnostics ();

    /** \brief Initialize IO buffer for beams
     *
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in] beamnames list of the names of the beam to be written to file
     */
    void InitBeamData (MultiBeam& beams, const amrex::Vector< std::string > beamnames);

    /** \brief writing openPMD data
     *
     * \param[in] field_diag field diagnostic data
     * \param[in] a_multi_beams multi beam container which is written to openPMD file
     * \param[in] physical_time Physical time of the currenerationt it.
     * \param[in] output_step current iteration to be written to file
     * \param[in] beamnames list of the names of the beam to be written to file
     * \param[in] geom3D 3D Geometry of the simulation, to get the cell size etc.
     * \param[in] call_type whether the beams or the fields should be written to file
     */
    void WriteDiagnostics (
        const amrex::Vector<FieldDiagnosticData>& field_diag, MultiBeam& a_multi_beam,
        const amrex::Real physical_time, const int output_step,
        const amrex::Vector< std::string > beamnames,
        amrex::Vector<amrex::Geometry> const& geom3D,
        const OpenPMDWriterCallType call_type);

    /** \brief Copy beam data into IO buffer
     *
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in] beamnames list of the names of the beam to be written to file
     */
    void CopyBeams (MultiBeam& beams, const amrex::Vector< std::string > beamnames);

    /** \brief Resets and flushes the openPMD series of all levels
     */
    void flush ();

    /** Prefix/path for the output files */
    std::string m_file_prefix;

    /** Temporary workaround to display normalized momentum correctly */
    bool m_openpmd_viewer_workaround = true;
};

#endif // HIPACE_USE_OPENPMD

#endif // OPENPMDWRITER_H_
