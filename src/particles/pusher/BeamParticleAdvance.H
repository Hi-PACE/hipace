/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef BEAMPARTICLEADVANCE_H_
#define BEAMPARTICLEADVANCE_H_

#include "particles/beam/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "Hipace.H"

/** Push beam particles contained in one z slice
 * \param[in,out] beam species of which the current is deposited
 * \param[in] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] current_N_level number of MR levels active on the current slice
 * \param[in] islice_local local index of the slice on which the beam particles are pushed
 */
void
AdvanceBeamParticlesSlice (BeamParticleContainer& beam, const Fields& fields,
                           amrex::Vector<amrex::Geometry> const& gm, int const current_N_level,
                           const int islice_local);

#endif //  BEAMPARTICLEADVANCE_H_
