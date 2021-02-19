#ifndef HIPACE_BinSort_H_
#define HIPACE_BinSort_H_

#include "BeamParticleContainer.H"

#include <AMReX_MultiFab.H>

/** \brief Find particles that are in each slice, and return collections of indices per slice.
 *
 * Note that this does *not* rearrange particle arrays
 *
 * \param[in] lev MR level
 * \param[in] ibox index of the box
 * \param[in] bx 3d box in which particles are sorted per slice
 * \param[in] beam Beam particle container
 * \param[in] geom Geometry
 */
amrex::DenseBins<BeamParticleContainer::ParticleType>
findParticlesInEachSlice (
    int lev, int ibox, amrex::Box bx,
    BeamParticleContainer& beam, amrex::Geometry& geom);

/** \brief Actually reorder the particles in memory according to the permutation array in a_bins.
 *
 * \param[in] a_beam Beam particle container
 * \param[in] a_bins amrex::DenseBins object constructed with a_beam
 */
void
reorderParticlesBySlice (
    BeamParticleContainer& a_beam,
    const amrex::DenseBins<BeamParticleContainer::ParticleType>& a_bins);


/** \brief Sort beam particles into the boxes in BoxArray a_ba.
 *
 * \param[in] a_beam Beam particle container
 * \param[in] a_ba BoxArray object to put the particles into
 * \param[in] a_geom Geometry object with the low corner of the domain
 */
void
sortParticlesByBox (
    BeamParticleContainer& a_beam,
    const amrex::BoxArray a_ba, const amrex::Geometry& a_geom);


#endif // HIPACE_BinSort_H_
