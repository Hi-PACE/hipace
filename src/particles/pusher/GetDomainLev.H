#ifndef HIPACE_GETDOMAINLEV_H_
#define HIPACE_GETDOMAINLEV_H_

#include <AMReX.H>

namespace
{
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> GetDomainLev (
        const amrex::Geometry& gm, const amrex::Box& box, bool is_lo, int lev)
    {
        if (lev == 0) {
            return is_lo ? gm.ProbLoArray() : gm.ProbHiArray();
        } else {
            const auto dx = gm.CellSizeArray();
            const amrex::IntVect& small = box.smallEnd();
            const amrex::IntVect& big = box.bigEnd();
            const auto plo = gm.ProbLoArray();
            if (is_lo) {
                return {{AMREX_D_DECL(plo[0]+(small[0]+0.5)*dx[0],
                                      plo[1]+(small[1]+0.5)*dx[1],
                                      plo[2]+(small[2]+0.5)*dx[2])}};
            } else {
                return {{AMREX_D_DECL(plo[0]+(big[0]+0.5)*dx[0],
                                      plo[1]+(big[1]+0.5)*dx[1],
                                      plo[2]+(big[2]+0.5)*dx[2])}};
            }
        }
    }
}

#endif // HIPACE_GETANDSETPOSITION_H_
