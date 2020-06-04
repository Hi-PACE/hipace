#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

class Hipace;

struct FieldComps{
    enum Comps{Ex = 0, Ey, Ez, jx, jy, jz, nfields};
};

class Fields
{
public:
    explicit Fields (Hipace const* a_hipace);

    void AllocData (int lev, const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm);

    amrex::Vector<amrex::MultiFab>& getF () { return m_F; }

    amrex::MultiFab& getF (int lev) { return m_F[lev]; }

    void CopySliceFtoS (int i_slice);
private:
    Hipace const* m_hipace;
    // Vector over levels, all fields
    amrex::Vector<amrex::MultiFab> m_F;
    // Number of slices in temporary object
    static constexpr int m_nslices = 4;
    // Vector over levels, array of 4 slices required to compute current slice
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>> m_slices;

    amrex::IntVect m_nguards {2, 2, 2};
    amrex::IntVect m_slices_nguards {2, 2, 0};
};

#endif
