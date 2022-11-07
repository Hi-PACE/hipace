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
#include "laser/Laser.H"

/** \brief Gather field values for particles, update the force terms, and push the particles
 * \param[in,out] plasma plasma species to push
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) will be used
 * \param[in] do_push boolean to define if plasma particles are pushed
 * \param[in] do_update boolean to define if the force terms are updated
 * \param[in] do_shift boolean to define if the force terms are shifted
 * \param[in] lev MR level
 * \param[in] bins objects containing indices of plasma particles in each tile
 * \param[in] laser laser pulse, which affects the plasma via the ponderomotive force
 */
void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, const Fields & fields,
                        amrex::Geometry const& gm, const bool temp_slice, const bool do_push,
                        const bool do_update, const bool do_shift, int const lev,
                        PlasmaBins& bins, const Laser& laser);

/** \brief Resets the particle position x, y, to x_prev, y_prev
 * \param[in,out] plasma plasma species to reset
 * \param[in] lev MR level
 * \param[in] initial whether this is initial reset at each time step. iF so, reset everything.
 */
void
ResetPlasmaParticles (PlasmaParticleContainer& plasma, int const lev, const bool initial=false);

#endif //  PLASMAPARTICLEADVANCE_H_
