/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef PLASMAPARTICLEADVANCE_H_
#define PLASMAPARTICLEADVANCE_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "particles/sorting/TileSort.H"
#include "fields/Fields.H"
#include "Hipace.H"
#include "laser/MultiLaser.H"

/** \brief Gather field values and push particles
 * \param[in,out] plasma plasma species to push
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) will be used
 * \param[in] lev MR level
 */
void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, const Fields & fields,
                        amrex::Vector<amrex::Geometry> const& gm, const bool temp_slice,
                        int const lev);

#endif //  PLASMAPARTICLEADVANCE_H_
