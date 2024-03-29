/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef PLASMADEPOSITCURRENT_H_
#define PLASMADEPOSITCURRENT_H_

#include "particles/plasma/PlasmaParticleContainer.H"
#include "particles/sorting/TileSort.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "laser/MultiLaser.H"
#include "Hipace.H"

/** Depose current of particles in species plasma into the current 2D slice in fields
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] multi_laser MultiLaser that affects the plasma during the deposition
 * \param[in] which_slice defines if this or the next slice is handled
 * \param[in] deposit_jx_jy if true, deposit to jx and jy
 * \param[in] deposit_jz if true, deposit to jz
 * \param[in] deposit_rho if true, deposit to rho
 * \param[in] deposit_chi if true, deposit chi
 * \param[in] deposit_rhomjz if true, deposit rhomjz
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 * \param[in] bins objects containing indices of plasma particles in each tile
 * \param[in] bin_size tile size (square)
 */
void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields, const MultiLaser& multi_laser,
                const int which_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                const bool deposit_chi, const bool deposit_rhomjz,
                amrex::Vector<amrex::Geometry> const& gm, int const lev,
                const PlasmaBins& bins, int bin_size);


#endif //  PLASMADEPOSITCURRENT_H_
