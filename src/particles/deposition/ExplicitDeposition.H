/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef EXPLICITDEPOSITION_H_
#define EXPLICITDEPOSITION_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"

/** Depose Sx and Sy of particles in species plasma into the current 2D slice in fields
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields,
                    amrex::Vector<amrex::Geometry> const& gm, const int lev);

#endif //  EXPLICITDEPOSITION_H_
