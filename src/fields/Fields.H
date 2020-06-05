#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

class Hipace;

struct FieldComps{
    enum Comps{ExmBy = 0, EypBx, Ez, Bx, By, Bz, jx, jy, jz, nfields};
};

enum struct FieldCopyType { FtoS, StoF };

struct Direction{
    enum dir{x=0, y, z};
};

class Fields
{
private:
    /** Number of slices in temporary slice object */
    static constexpr int m_nslices = 4;

public:
    explicit Fields (Hipace const* a_hipace);

    void AllocData (int lev, const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm);

    amrex::Vector<amrex::MultiFab>& getF () { return m_F; }

    amrex::MultiFab& getF (int lev) { return m_F[lev]; }

    amrex::Vector<std::array<amrex::MultiFab, m_nslices>>& getSlices () {return m_slices; }

    std::array<amrex::MultiFab, m_nslices>& getSlices (int lev) {return m_slices[lev]; }

    amrex::MultiFab& getSlices (int lev, int i) {return m_slices[lev][i]; }

    /** Copy between the full MultiFab and slice MultiFab */
    void Copy (int lev, int i_slice, FieldCopyType copy_type, int slice_comp, int full_comp,
               int ncomp);

    void ShiftSlices (int lev);

    /** Compute transverse derivative of 1 slice*/
    void TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                               const amrex::Real dx, const int scomp=0, const int dcomp=0);

private:
    Hipace const* m_hipace;
    /** Vector over levels, all fields */
    amrex::Vector<amrex::MultiFab> m_F;
    /** Vector over levels, array of 4 slices required to compute current slice */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>> m_slices;
    /** Number of guard cells for main MultiFab */
    amrex::IntVect m_nguards {2, 2, 2};
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards {2, 2, 0};
};

#endif
