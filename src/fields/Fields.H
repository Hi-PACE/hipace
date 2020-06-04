#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

struct FieldComps{
    enum Comps{ExmBy = 0, EypBx, Ez, Bx, By, Bz, jx, jy, jz, nfields};
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
private:
    // Vector over levels, all fields
    amrex::Vector<amrex::MultiFab> m_F;
    // Number of slices in temporary object
    static constexpr int m_nslices = 4;
    // Vector over levels, array of 4 slices required to compute current slice
    amrex::Vector<std::array<amrex::MultiFab*, m_nslices>> m_slices;

    amrex::IntVect m_nguards {2, 2, 2};
    amrex::IntVect m_slices_nguards {2, 2, 0};
};

#endif
