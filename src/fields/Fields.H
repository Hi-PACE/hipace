#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

struct FieldComps{
    enum Comps{Ex = 0, Ey, Ez, jx, jy, jz, nfields};
};

class Fields
{
public:
    explicit Fields (int max_level) : m_F(max_level+1) {}

    void AllocData (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

    amrex::Vector<amrex::MultiFab>& getF () { return m_F; }

    amrex::MultiFab& getF (int lev) { return m_F[lev]; }
private:
    amrex::Vector<amrex::MultiFab> m_F;
    amrex::IntVect m_nguards {2, 2, 2};
};

#endif
