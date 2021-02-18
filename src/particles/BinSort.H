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

/** \brief Reorder particles such that those in the same slice are adjacent in memory.
 *
 * \param[in] a_beam Beam particle container
 * \param[in] a_bins amrex::DenseBins object constructed with a_beam
 */
void
reorderParticlesInEachSlice (
    BeamParticleContainer& a_beam,
    const amrex::DenseBins<BeamParticleContainer::ParticleType>& a_bins);

#endif // HIPACE_BinSort_H_
