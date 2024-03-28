/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
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

/** Whether to call amrex::Gpu::streamSynchronize() around all profiler region */
inline int DO_DEVICE_SYNCHRONIZE = 0;

template<int detail_level>
AMREX_FORCE_INLINE
void doStreamSynchronize ()
{
    if ( DO_DEVICE_SYNCHRONIZE >= detail_level )
        amrex::Gpu::streamSynchronize();
}

// Note that objects are destructed in the reverse order of declaration
template<int detail_level>
struct synchronizeOnDestruct {
    AMREX_FORCE_INLINE
    ~synchronizeOnDestruct () {
        doStreamSynchronize<detail_level>();
    }
};

#define HIPACE_PROFILE(fname) doStreamSynchronize<1>(); BL_PROFILE(fname); synchronizeOnDestruct<1> BL_PROFILE_PASTE(SYNC_SCOPE_, __COUNTER__){}
#define HIPACE_PROFILE_VAR(fname, vname) doStreamSynchronize<1>(); BL_PROFILE_VAR(fname, vname); synchronizeOnDestruct<1> SYNC_V_##vname{}
#define HIPACE_PROFILE_VAR_NS(fname, vname) BL_PROFILE_VAR_NS(fname, vname); synchronizeOnDestruct<1> SYNC_V_##vname{}
#define HIPACE_PROFILE_VAR_START(vname) doStreamSynchronize<1>(); BL_PROFILE_VAR_START(vname)
#define HIPACE_PROFILE_VAR_STOP(vname) doStreamSynchronize<1>(); BL_PROFILE_VAR_STOP(vname)
#define HIPACE_PROFILE_REGION(rname) doStreamSynchronize<1>(); BL_PROFILE_REGION(rname); synchronizeOnDestruct<1> BL_PROFILE_PASTE(SYNC_R_, __COUNTER__){}

#endif // HIPACE_PROFILERWRAPPER_H_
