/* Copyright 2020-2022 AlexanderSinn, Axel Huebl, MaxThevenet
 * Weiqun Zhang
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        HIPACE_PROFILE("main()");
        Hipace hipace;
        hipace.InitData();
        hipace.Evolve();
    }
    amrex::Finalize();
}
