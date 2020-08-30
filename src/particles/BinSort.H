#ifndef HIPACE_BinSort_H_
#define HIPACE_BinSort_H_

#include "BeamParticleContainer.H"

#include <AMReX_MultiFab.H>

using ParticleType = BeamParticleContainer::ParticleType;
using ParticleTileType = BeamParticleContainer::ParticleTileType;
using ParticleBins = amrex::DenseBins<ParticleType>;
using index_type = ParticleBins::index_type;

/** Find the particles and count the particles that are in each cell.
 * Note that this does *not* rearrange particle arrays */
ParticleBins
findParticlesInEachCell( int const lev, amrex::Box bx,
                         ParticleTileType const& ptile, amrex::Geometry geom);

#endif // HIPACE_BinSort_H_
