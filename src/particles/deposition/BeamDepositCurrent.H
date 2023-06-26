/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Remi Lehe, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef BEAMDEPOSITCURRENT_H_
#define BEAMDEPOSITCURRENT_H_

#include "particles/beam/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "particles/sorting/SliceSort.H"

/** Depose current of beam particles on a single slice
 * \param[in] beam species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 * \param[in] do_beam_jx_jy_deposition whether the beams deposit Jx and Jy
 * \param[in] do_beam_jz_deposition whether the beams deposit Jz
 * \param[in] do_beam_rhomjz_deposition whether the beams deposit rhomjz
 * \param[in] which_slice defines if this, next the salame field slice is used
 * \param[in] which_beam_slice defines if this or the next beam slice is used
 * \param[in] only_highest if the particles deposit only on their highest MR level
 */
void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Vector<amrex::Geometry> const& gm, int const lev,
                     const bool do_beam_jx_jy_deposition,
                     const bool do_beam_jz_deposition,
                     const bool do_beam_rhomjz_deposition,
                     const int which_slice, const WhichBeamSlice::beam_slice which_beam_slice,
                     const bool only_highest=false);


#endif //  BEAMDEPOSITCURRENT_H_
