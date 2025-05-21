/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: EyaDammak, AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef TEMPERATUREDEPOSITION_H_
#define TEMPERATUREDEPOSITION_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"

/** Depose current of particles in species plasma into the current 2D slice in fields
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
DepositTemperature (PlasmaParticleContainer& plasma, Fields & fields,
                amrex::Vector<amrex::Geometry> const& gm, int const lev);


#endif //  TEMPERATUREDEPOSITION_H_
