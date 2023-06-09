/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef BEAMPARTICLEADVANCE_H_
#define BEAMPARTICLEADVANCE_H_

#include "particles/beam/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "Hipace.H"

/** Push beam particles contained in one z slice
 * \param[in,out] beam species of which the current is deposited
 * \param[in] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] current_N_level number of MR levels active on the current slice
 * \param[in] islice_local local index of the slice on which the beam particles are pushed
 * \param[in] extEu uniform external electric field applied to beam particles.
              The components represent Ex-c*By, Ey+c*Bx and Ez respectively.
 * \param[in] extBu Uniform external magnetic field applied to beam particles.
              The components represent Bx, By and Bz, respectively.
 * \param[in] extEs Slope of a linear external electric field applied to beam particles.
              The components represent d(Ex-c*By)/dx, d(Ey+c*Bx)/dy and d(Ez)/dz respectively.
              For the last component, z represents the zeta coordinate zeta = z - c*t.
 * \param[in] extBs Slope of a linear external magnetic field applied to beam particles.
              The components represent d(Bx)/dy, d(By)/dx and d(Bz)/dz respectively.
              Note the order of derivatives for the transverse components!
              For the last component, z represents the zeta coordinate zeta = z - c*t
 */
void
AdvanceBeamParticlesSlice (BeamParticleContainer& beam, const Fields& fields,
                           amrex::Vector<amrex::Geometry> const& gm, int const current_N_level,
                           const int islice_local, const amrex::RealVect& extEu,const amrex::RealVect& extBu,const amrex::RealVect& extEs,const amrex::RealVect& extBs);

#endif //  BEAMPARTICLEADVANCE_H_
