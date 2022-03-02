/* Copyright 2020 MaxThevenet, Remi Lehe, WeiqunZhang
 *
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolver.H"

FFTPoissonSolver::~FFTPoissonSolver ()
{}

amrex::MultiFab&
FFTPoissonSolver::StagingArea ()
{
    return m_stagingArea;
}
