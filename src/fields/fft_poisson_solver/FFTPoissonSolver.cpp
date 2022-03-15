/* Copyright 2020
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Remi Lehe, WeiqunZhang
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
