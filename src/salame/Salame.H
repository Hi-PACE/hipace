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
 * \param[in,out] overloaded if the SALAME beam was overloaded in the last slice
 * \param[in] current_N_level number of MR levels active on the current slice
 * \param[in] step time step of simulation
 * \param[in] islice slice index of the whole domain
 * \param[in] relative_tolerance relative error tolerance to finish SALAME iterations early
 */
void
SalameModule (Hipace* hipace, const int n_iter, const bool do_advance, int& last_islice,
              bool& overloaded, const int current_N_level, const int step, const int islice,
              const amrex::Real relative_tolerance);

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
 * \param[in] current_N_level number of MR levels active on the current slice
 * \param[in] islice slice index of the whole domain
 * \return new beam weighting factor and new total SALAME beam current on this slice
 */
std::pair<amrex::Real, amrex::Real>
SalameGetW (Hipace* hipace, const int current_N_level, const int islice);

/** Multiply SALAME beam weight on this slice with W
 * \param[in] W weight multiplier
 * \param[in] hipace pointer to Hipace instance
 */
void
SalameMultiplyBeamWeight (const amrex::Real W, Hipace* hipace);

#endif //  SALAME_H_
