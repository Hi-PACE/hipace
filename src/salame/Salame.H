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

/** Depose Sx and Sy of particles in species plasma into the current 2D slice in fields
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] laser that affects the plasma during the deposition
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
SalameModule (Hipace* hipace, const int lev, const int islice);

#endif //  SALAME_H_
