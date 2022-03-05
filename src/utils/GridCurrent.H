/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef GRIDCURRENT_H_
#define GRIDCURRENT_H_

#include "fields/Fields.H"
#include <AMReX_AmrCore.H>

/** \brief class handling a current directly written to the grid */
class GridCurrent
{
private:

    /** Whether to use a grid current */
    bool m_use_grid_current = false;
    /** Average position of the Gaussian grid current. */
    amrex::RealVect m_position_mean {0., 0., 0.};
    /** Width of the Gaussian grid current. */
    amrex::RealVect m_position_std {0., 0., 0.};
    amrex::Real m_peak_current_density {0.}; /**< peak density for the grid current */
    int m_finest_level {0}; /**< finest level of mesh refinement that the beam interacts with */

public:
    /** Constructor */
    explicit GridCurrent ();

    /** calculate the adaptive time step based on the beam energy
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     * \param[in] islice index of the slice on which the beam particles are pushed
     */
    void
    DepositCurrentSlice (Fields& fields, const amrex::Geometry& geom, int const lev,
                         const int islice);

};

#endif // GRIDCURRENT_H_
