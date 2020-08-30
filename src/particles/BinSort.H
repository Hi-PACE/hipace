#ifndef HIPACE_BinSort_H_
#define HIPACE_BinSort_H_

#include "BeamParticleContainer.H"

#include <AMReX_MultiFab.H>

using ParticleType = BeamParticleContainer::ParticleType;
using ParticleTileType = BeamParticleContainer::ParticleTileType;
using ParticleBins = amrex::DenseBins<ParticleType>;
using index_type = ParticleBins::index_type;

/** \brief Find the particles and count the particles that are in each cell.
 * Note that this does *not* rearrange particle arrays
 *
 * \param[in] lev MR level
 * \param[in] bx 3d box in which particles are sorted per slice
 * \param[in] ptile particle data for particles in bx
 * \param[in] geom Geometry
 */
ParticleBins
findParticlesInEachCell (amrex::Box bx, BeamParticleContainer::ParticleTileType const& ptile,
                         amrex::Geometry geom);

#endif // HIPACE_BinSort_H_
