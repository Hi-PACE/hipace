/* Copyright 2026
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HISTOGRAMDEPOSITION_H_
#define HISTOGRAMDEPOSITION_H_

#include "particles/beam/BeamParticleContainer.H"
#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "diagnostics/Diagnostic.H"

/** Make a histogram of plasma particles
 * \param[in] plasma species of which the histogram is deposited
 * \param[in,out] fd diagnostic data
 * \param[in] zmid z position of the slice
 */
void
HistogramDepositionPlasma (PlasmaParticleContainer& plasma, DiagnosticData& fd, amrex::Real zmid);

/** Make a histogram of beam particles
 * \param[in] beam species of which the histogram is deposited
 * \param[in,out] fd diagnostic data
 */
void
HistogramDepositionBeam (BeamParticleContainer& beam, DiagnosticData& fd);


#endif //  HISTOGRAMDEPOSITION_H_
