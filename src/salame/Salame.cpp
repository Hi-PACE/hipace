/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "Salame.H"

void
SalameModule (Hipace* hipace, const int lev, const int islice)
{
    hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_target"}
                                    WhichSlice::This, {"Ez"});

    hipace->m_multi_plasma.AdvanceParticles(hipace->m_fields, hipace->m_laser, hipace->geom[lev],
                                            true, true, true, true, lev);

    hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"jx", "jy"}
                                    WhichSlice::Next, {"jx_beam"}, {"jy_beam"});

    hipace->m_multi_plasma.DepositCurrent(hipace->m_fields, hipace->m_laser,
            WhichSlice::Salame, true, true, false, false, false, hipace-> geom[lev], lev);

    hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jz_beam", "Sy", "Sx");

    hipace->m_fields.SolvePoissonEz(hipace->Geom(), lev, islice, WhichSlice::Salame);

    hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_no_salame"}
                                    WhichSlice::Salame, {"Ez"});

    hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->geom, lev, step,
        islice_local, beam_bin, hipace->m_box_sorters, ibox, false, true, false, WhichSlice::Salame);

    // get SxSy from jz_beam

    hipace->ExplicitMGSolveBxBy(lev, WhichSlice::Salame);

    hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jz_beam", "jx", "jy");

    // get jx and jy from Bx, By and chi

    hipace->m_fields.SolvePoissonEz(hipace->Geom(), lev, islice, WhichSlice::Salame);

    // get W with average (Ez_target - Ez_no_salame) / Ez

    // multiply salame beam weight with W

    hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->geom, lev, step,
        islice_local, beam_bin, hipace->m_box_sorters, ibox, false, true, false, WhichSlice::Salame);

    // add SxSy (This) from jz_beam (Salame)

    hipace->ExplicitMGSolveBxBy(lev, WhichSlice::This);
};
