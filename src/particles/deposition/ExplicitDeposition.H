/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef EXPLICITDEPOSITION_H_
#define EXPLICITDEPOSITION_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"

void
ExplicitDeposition (PlasmaParticleContainer& plasma, Fields& fields,
                    amrex::Geometry const& gm, const int lev);


#endif //  EXPLICITDEPOSITION_H_
