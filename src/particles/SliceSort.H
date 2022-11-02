/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_SLICESORT_H_
#define HIPACE_SLICESORT_H_

#include "BeamParticleContainer.H"
#include "BoxSort.H"

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
 * \param[in] a_box_sorter object that sorts particles by box
 */
BeamBins
findParticlesInEachSlice (
    int lev, int ibox, amrex::Box bx, BeamParticleContainer& beam,
    amrex::Vector<amrex::Geometry> const& geom, const BoxSorter& a_box_sorter);

#endif // HIPACE_SLICESORT_H_
