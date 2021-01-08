#ifndef FIELDDIAGNOSTIC_H_
#define FIELDDIAGNOSTIC_H_

#include <AMReX_MultiFab.H>

enum struct DiagType{xyz, xz, yz};

class FieldDiagnostic
{
public:
    explicit FieldDiagnostic (int nlev);
    void AllocData (int lev, const amrex::BoxArray& ba, int nfields, const amrex::DistributionMapping& dm, amrex::Geometry const& geom);
    amrex::Vector<amrex::MultiFab>& getF () { return m_F; };
    amrex::MultiFab& getF (int lev) {return m_F[lev];};
    amrex::Vector<amrex::Geometry>& getGeom () { return m_geom_io; };
    int sliceDir () {return m_slice_dir;};
private:
    /** Vector over levels, all fields */
    amrex::Vector<amrex::MultiFab> m_F;
    DiagType m_diag_type;
    int m_slice_dir;
    int m_nfields;
    amrex::Vector<amrex::Geometry> m_geom_io;
};

#endif // FIELDDIAGNOSTIC_H_
