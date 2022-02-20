#ifndef HIPACE_TILESORT_H_
#define HIPACE_TILESORT_H_

#include "PlasmaParticleContainer.H"

#include <AMReX_MultiFab.H>

using PlasmaBins = amrex::DenseBins<PlasmaParticleContainer::ParticleType>;

/** \brief Find plasma particles in each bin, and return collections of indices per bin (tile).
 *
 * Note that this does *not* rearrange particle arrays
 *
 * \param[in] lev MR level
 * \param[in] bx 3d box in which particles are sorted per slice
 * \param[in] bin_size number of cells per tile (square)
 * \param[in] plasma Plasma particle container
 * \param[in] geom Geometry
 */
PlasmaBins
findParticlesInEachTile (
    int lev, amrex::Box bx, int bin_size,
    PlasmaParticleContainer& plasma, const amrex::Geometry& geom);

#endif // HIPACE_TILESORT_H_
