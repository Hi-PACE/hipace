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
    amrex::Vector<std::string> m_names {"no_gridplasma"};

public:
    /** Read parameters from the input file */
    void ReadParameters ();

    std::vector<std::string> GetFieldComponents (const MultiPlasma& multi_plasma);

    void InitData (Fields& fields, const MultiPlasma& multi_plasma,
                   const amrex::Geometry& geom, int const lev);

    /** calculate the adaptive time step based on the beam energy
     * \param[in,out] fields the general field class, modified by this function
     * \param[in] geom Geometry of the simulation, to get the cell size etc.
     * \param[in] lev MR level
     * \param[in] islice index of the slice on which the beam particles are pushed
     */
    void
    IonizeGrid (Fields& fields, const MultiPlasma& multi_plasma, const MultiLaser& laser,
                const amrex::Geometry& geom, int const lev,
                const amrex::Geometry& laser_geom);

};

#endif // GRIDIONIZATION_H_
