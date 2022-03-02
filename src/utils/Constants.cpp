/* Copyright 2020 MaxThevenet
 *
 * This file is part of HiPACE++.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Constants.H"
#include "Hipace.H"

PhysConst get_phys_const ()
{
    Hipace& hipace = Hipace::GetInstance();
    return hipace.get_phys_const ();
}
