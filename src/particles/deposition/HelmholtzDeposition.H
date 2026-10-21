/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HELMHOLTZDEPOSITION_H_
#define HELMHOLTZDEPOSITION_H_

#include "particles/beam/BeamParticleContainer.H"
#include "helmholtz/Helmholtz.H"
#include "utils/Constants.H"

/** Depose beam currents for into Helmholtz current arrays
 * \param[in] beam Beam species that is deposited in array
 * \param[in] helmholtz Helmholtz object into which currents are deposited
 * \param[in] which_beam_slice Beam slice to depose.
 * \param[in] time Current physical time in the simulation.
 */
void
HelmholtzDeposition (BeamParticleContainer& beam, Helmholtz& helmholtz,
                     const int which_beam_slice, amrex::Real time);

#endif //  HELMHOLTZDEPOSITION_H_
