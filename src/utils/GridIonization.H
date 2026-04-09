/* Copyright 2026
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef GRIDIONIZATION_H_
#define GRIDIONIZATION_H_

#include "fields/Fields.H"
#include "particles/profiles/GetInitialDensity.H"
#include "particles/plasma/MultiPlasma.H"
#include <AMReX_AmrCore.H>

/** \brief class handling ionization directly written to the grid */
class GridIonization
{
private:

    /** Whether to use a grid ionization */
    bool m_use_grid_ionization = false;
    /** Names of plasma species used in grid ionization */
    amrex::Vector<std::string> m_names {"no_gridplasma"};

public:
    /** Read parameters from the input file */
    void ReadParameters ();

    /** Return a vector of component names that need to be used by grid ionization */
    std::vector<std::string> GetFieldComponents (const MultiPlasma& multi_plasma);

    /** Initialize gridion fields. Must be called after Plasma initialize for every time step
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] multi_plasma plasma object
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     */
    void InitData (Fields& fields, const MultiPlasma& multi_plasma,
                   const amrex::Geometry& geom, int const lev);

    /** calculate laser ionization using grid quantaties
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] multi_plasma plasma object
     * \param[in] laser laser object
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     * \param[in] laser_geom geometry of the laser
     */
    void
    IonizeGrid (Fields& fields, const MultiPlasma& multi_plasma, const MultiLaser& laser,
                const amrex::Geometry& geom, int const lev,
                const amrex::Geometry& laser_geom);

};

#endif // GRIDIONIZATION_H_
