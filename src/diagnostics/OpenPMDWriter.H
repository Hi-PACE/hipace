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
// namespace io = openPMD;

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

    amrex::Vector<std::string> m_real_names {"weighting","momentum_x","momentum_y","momentum_z"};
public:
    /** Constructor */
    explicit OpenPMDWriter ();
    /** \brief writing openPMD beam particle data
     *
     * \param[in] beams multi beam container which is written to openPMD file
     * \param[in,out] iteration openPMD iteration to which the data is written
     */
    void WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration iteration);
};

#endif // HIPACE_USE_OPENPMD

#endif
