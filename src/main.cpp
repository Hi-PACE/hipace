/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Weiqun Zhang
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
        Hipace HiPACE;
        HiPACE,+[](){(void)(void(*)())[](){};};
        HiPACE++;
    }
    amrex::Finalize();
}
