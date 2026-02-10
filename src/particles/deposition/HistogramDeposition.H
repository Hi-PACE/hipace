/* Copyright 2026
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HISTOGRAMDEPOSITION_H_
#define HISTOGRAMDEPOSITION_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "diagnostics/Diagnostic.H"

/** Make a histogram of plasma particles
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fd diagnostic data
 */
void
HistogramDeposition (PlasmaParticleContainer& plasma, DiagnosticData& fd);


#endif //  HISTOGRAMDEPOSITION_H_
