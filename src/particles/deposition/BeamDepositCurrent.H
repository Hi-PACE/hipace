/* Copyright 2020-2022 AlexanderSinn, MaxThevenet, Remi Lehe
 * Severin Diederichs
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef BEAMDEPOSITCURRENT_H_
#define BEAMDEPOSITCURRENT_H_

#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "particles/SliceSort.H"

/** Depose current of beam particles on a single slice
 * \param[in] beam species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 * \param[in] islice index of the slice on which the beam particles are pushed
 * \param[in] offset offset to the particles on the current box
 * \param[in] bins beam particle container bins, to push only the beam particles on slice islice
 * \param[in] do_beam_jx_jy_deposition whether the beams deposit Jx and Jy
 * \param[in] which_slice defines if this or the next slice is handled
 * \param[in] nghost number of ghost particles, all at the end of the particle array.
 *            Use for depositing transverse currents in the Next slice when processing
 *            islice = 0.
 */
void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Vector<amrex::Geometry> const& gm, int const lev, const int islice,
                     int const offset, const BeamBins& bins,
                     const bool do_beam_jx_jy_deposition, const int which_slice,
                     int nghost=0);


#endif //  BEAMDEPOSITCURRENT_H_
