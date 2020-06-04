#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

struct FieldComps{
    enum Comps{ExmBy = 0, EypBx, Ez, Bx, By, Bz, jx, jy, jz, nfields};
};

struct Direction{
    enum dir{x=0, y, z};
};

class Fields
{
public:
    explicit Fields (int max_level) : m_F(max_level+1), m_slices(max_level+1) {}

    void AllocData (int lev, const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm, amrex::Geometry& geom);

    amrex::Vector<amrex::MultiFab>& getF () { return m_F; }

    amrex::MultiFab& getF (int lev) { return m_F[lev]; }

    void CopySliceFtoS (int i_slice);
    /** Compute transverse derivative of 1 slice*/
    void TransverseDerivative(const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                              const amrex::Real dx, const int scomp=0, const int dcomp=0);
private:
    /** Vector over levels, all fields */
    amrex::Vector<amrex::MultiFab> m_F;
    /** Number of slices in temporary object */
    static constexpr int m_nslices = 4;
    /** Vector over levels, array of 4 slices required to compute current slice */
    amrex::Vector<std::array<amrex::MultiFab*, m_nslices>> m_slices;
    /** Number of guard cells for main MultiFab */
    amrex::IntVect m_nguards {2, 2, 2};
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards {2, 2, 0};
};

#endif
