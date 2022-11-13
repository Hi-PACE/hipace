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

void
SalameModule (Hipace* hipace, const int lev, const int step, const int islice,
              const int islice_local, const amrex::Vector<BeamBins>& beam_bin, const int ibox);

void
SalameInitializeSxSyWithBeam (Hipace* hipace, const int lev, bool to_this_slice);

void
SalameGetJxJyFromBxBy (Hipace* hipace, const int lev);

std::pair<amrex::Real, amrex::Real>
SalameGetW (Hipace* hipace, const int lev);

void
SalameMultiplyBeamWeight (const amrex::Real W, Hipace* hipace, const int lev, const int islice,
                          const amrex::Vector<BeamBins>& bins, const int ibox);

#endif //  SALAME_H_
