/* Copyright 2020 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef HIPACE_PROFILERWRAPPER_H_
#define HIPACE_PROFILERWRAPPER_H_

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuDevice.H>

AMREX_FORCE_INLINE
void doDeviceSynchronize(int do_device_synchronize)
{
    if ( do_device_synchronize )
        amrex::Gpu::synchronize();
}

#define HIPACE_PROFILE(fname) BL_PROFILE(fname)
#define HIPACE_PROFILE_VAR(fname, vname) BL_PROFILE_VAR(fname, vname)
#define HIPACE_PROFILE_VAR_NS(fname, vname) BL_PROFILE_VAR_NS(fname, vname)
#define HIPACE_PROFILE_VAR_START(vname) BL_PROFILE_VAR_START(vname)
#define HIPACE_PROFILE_VAR_STOP(vname) BL_PROFILE_VAR_STOP(vname)
#define HIPACE_PROFILE_REGION(rname) BL_PROFILE_REGION(rname)

#endif // HIPACE_PROFILERWRAPPER_H_
