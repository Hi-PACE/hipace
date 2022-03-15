/* Copyright 2020
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#include "Constants.H"
#include "Hipace.H"

PhysConst get_phys_const ()
{
    Hipace& hipace = Hipace::GetInstance();
    return hipace.get_phys_const ();
}
