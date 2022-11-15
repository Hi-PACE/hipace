/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef SALAME_H_
#define SALAME_H_

#include "Hipace.H"
#include "utility"

/** Calculate new weight for this slice of SALAME beams and recompute the effected fields
 * \param[in] hipace pointer to Hipace instance
 * \param[in] n_iter the number of SALAME iterations to be done
 * \param[in] do_advance if the SALAME-only field should be computed exactly with plasma particles
 * \param[in,out] last_islice the slice index of the previous slice with SALAME
 * \param[in] lev MR level
 * \param[in] step time step of simulation
 * \param[in] islice slice index of the whole domain
 * \param[in] islice slice index of the local box
 * \param[in] beam_bin Vector (over species) of beam particles sorted by slices
 * \param[in] ibox index of the current box
 */
void
SalameModule (Hipace* hipace, const int n_iter, const bool do_advance, int& last_islice,
              const int lev, const int step, const int islice, const int islice_local,
              const amrex::Vector<BeamBins>& beam_bin, const int ibox);

/** Initialize Sx and Sy with the contribution from the SALAME beam
 * \param[in] hipace pointer to Hipace instance
 * \param[in] lev MR level
 */
void
SalameInitializeSxSyWithBeam (Hipace* hipace, const int lev);

/** Approximate the extra jx and jy from the SALAME beam using Bx By and chi
 * \param[in] hipace pointer to Hipace instance
 * \param[in] lev MR level
 */
void
SalameGetJxJyFromBxBy (Hipace* hipace, const int lev);

/** Advance plasma particles to the temp slice using only Bx and By from the SALAME beam
 * \param[in] hipace pointer to Hipace instance
 * \param[in] lev MR level
 */
void
SalameOnlyAdvancePlasma (Hipace* hipace, const int lev);

/** Calculate the new weighting factor of the SALAME beam using the difference in E fields.
 * The average is weighted using the SALAME beam current.
 * \param[in] hipace pointer to Hipace instance
 * \param[in] lev MR level
 * \return new beam weighting factor and new total SALAME beam current on this slice
 */
std::pair<amrex::Real, amrex::Real>
SalameGetW (Hipace* hipace, const int lev);

/** Multiply SALAME beam weight on this slice with W
 * \param[in] weight multiplier
 * \param[in] hipace pointer to Hipace instance
 * \param[in] islice slice index of the local box
 * \param[in] bins Vector (over species) of beam particles sorted by slices
 * \param[in] ibox index of the current box
 */
void
SalameMultiplyBeamWeight (const amrex::Real W, Hipace* hipace, const int islice,
                          const amrex::Vector<BeamBins>& bins, const int ibox);

#endif //  SALAME_H_
