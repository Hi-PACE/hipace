/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef OPENPMDWRITER_H_
#define OPENPMDWRITER_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_AmrCore.H>

#include <cstdint>
#include <vector>

namespace openPMD {
    class Series;
    class Iteration;
    class ParticleSpecies;
    class Record;
}

struct DiagnosticData;
class Fields;
class MultiBeam;
class MultiLaser;
class MultiPlasma;

#ifdef HIPACE_USE_OPENPMD
/** \brief class handling the IO with openPMD */
class OpenPMDWriter
{
private:

    /** \brief writing openPMD field data
     *
     * \param[in] fd diagnostic data
     * \param[in] multi_laser multi laser to get the central wavelength
     * \param[in,out] iteration openPMD iteration to which the data is written
     */
    void WriteFieldData (const DiagnosticData& fd, const MultiLaser& multi_laser,
                         openPMD::Iteration& iteration);

    /** \brief writing openPMD particle data
     *
     * \param[in] fd diagnostic data
     * \param[in,out] iteration openPMD iteration to which the data is written
     * \param[in] beams multi beam to get charge and mass
     * \param[in] plasmas multi plamsa to get charge and mass
     * \param[in] geom 3D Geometry of the simulation, to get the cell size
     */
    void WriteParticleData (DiagnosticData& fd, openPMD::Iteration& iteration,
                            const MultiBeam& beams, const MultiPlasma& plasmas,
                            const amrex::Geometry& geom);

    /** \brief setup the openPMD parameters do dump particle data
     *
     * \param[in] species_name name of the particle species
     * \param[in,out] particle_species openPMD species to set up
     * \param[in] np_total total number of particles in the bunch
     * \param[in] beams multi beam to get charge and mass
     * \param[in] plasmas multi plamsa to get charge and mass
     * \param[in] geom 3D Geometry of the simulation, to get the cell size
     * \param[in] is_plama_slice don't set z component of metadata
     */
    void SetupAttributes (const std::string& species_name,
                          openPMD::ParticleSpecies& particle_species, std::size_t np_total,
                          const MultiBeam& beams, const MultiPlasma& plasmas,
                          const amrex::Geometry& geom, bool only_xy);

    /** \brief setup the openPMD parameters do dump particle data
     *
     * \param[in,out] particle_record openPMD record to set up
     * \param[in] currRecord name of the record
     * \param[in,out] addedRecords records that where already setup
     */
    void SetupRecord (openPMD::Record particle_record, const std::string& currRecord,
                      std::set<std::string>& addedRecords);

    /** vector over levels of openPMD-api Series object for output */
    std::unique_ptr< openPMD::Series > m_outputSeries;

    /** openPMD backend: h5, bp, or json. Default depends on what is available */
    std::string m_openpmd_backend = "default";

public:
    /** Constructor */
    explicit OpenPMDWriter ();

    ~OpenPMDWriter ();

    /** Read parameters from the input file */
    void ReadParameters ();

    /** \brief Initialize diagnostics (collective operation)
     */
    void InitDiagnostics ();

    /** \brief writing openPMD particle and field data
     *
     * \param[in] diag_data diagnostic data
     * \param[in] multi_laser multi laser to get lambda0
     * \param[in] beams multi beam to get charge and mass
     * \param[in] plasmas multi plamsa to get charge and mass
     * \param[in] geom 3D Geometry of the simulation, to get the cell size
     * \param[in] physical_time Physical time of the current iteration.
     * \param[in] output_step current iteration to be written to file
     */
    void WriteDiagnostics (
        amrex::Vector<DiagnosticData>& diag_data,
        const MultiLaser& multi_laser, const MultiBeam& beams, const MultiPlasma& plasmas,
        const amrex::Geometry& geom, const amrex::Real physical_time, const int output_step);

    /** Prefix/path for the output files */
    std::string m_file_prefix;
};

#endif // HIPACE_USE_OPENPMD

#endif // OPENPMDWRITER_H_
