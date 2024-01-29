/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MGPoissonSolverDirichlet.H"
#include "fields/Fields.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

MGPoissonSolverDirichlet::MGPoissonSolverDirichlet (
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    m_stagingArea = amrex::MultiFab(ba, dm, 1, Fields::m_poisson_nguards);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ba.size() == 1, "Parallel MG not supported");
    amrex::Box solve_box = ba[0].grow(Fields::m_poisson_nguards);
    m_mg = std::make_unique<hpmg::MultiGrid>(gm.CellSize(0), gm.CellSize(1), solve_box, 3);
}

void
MGPoissonSolverDirichlet::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("MGPoissonSolverDirichlet::SolvePoissonEquation()");

    for ( amrex::MFIter mfi(m_stagingArea, DfltMfi); mfi.isValid(); ++mfi ){
        const int max_iters = 200;
        m_mg->solve3(lhs_mf[mfi], m_stagingArea[mfi], m_MG_tolerance_rel, m_MG_tolerance_abs,
                     max_iters, m_MG_verbose);
    }
}
