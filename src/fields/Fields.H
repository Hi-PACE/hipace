#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

struct FieldComps{
    enum Comps{Ex = 0, Ey, Ez, nfields};
};

class Fields
{
public:
    Fields (int max_level) : m_F(max_level) {};
    void AllocData (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
private:
    amrex::Vector<amrex::MultiFab> m_F;
    amrex::IntVect m_nguards {2, 2, 2};
};

#endif
