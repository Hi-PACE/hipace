/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_TILESORT_H_
#define HIPACE_TILESORT_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "particles/beam/BeamParticleContainer.H"

#include <AMReX_MultiFab.H>

using PlasmaBins = amrex::DenseBins<PlasmaParticleContainer::ParticleTileType::ParticleTileDataType>;
using BeamBins = amrex::DenseBins<BeamTile::ParticleTileDataType>;

/** \brief Find plasma particles in each bin, and return collections of indices per bin (tile).
 *
 * Note that this does *not* rearrange particle arrays
 *
 * \param[in] bx 3d box in which particles are sorted per slice
 * \param[in] bin_size number of cells per tile (square)
 * \param[in] plasma Plasma particle container
 * \param[in] geom Geometry
 */
PlasmaBins
findParticlesInEachTile (
    amrex::Box bx, int bin_size,
    PlasmaParticleContainer& plasma, const amrex::Geometry& geom);

/** \brief Find beam particles in each bin, and return collections of indices per bin (tile).
 *
 * Note that this does *not* rearrange particle arrays
 *
 * \param[in] bx 3d box in which particles are sorted per slice
 * \param[in] bin_size number of cells per tile (square)
 * \param[in] beam beam particle container
 * \param[in] geom Geometry
 */
BeamBins
findBeamParticlesInEachTile (
    amrex::Box bx, int bin_size,
    BeamParticleContainer& beam, const amrex::Geometry& geom);

#endif // HIPACE_TILESORT_H_
