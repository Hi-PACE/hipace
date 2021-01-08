#ifndef FIELDDIAGNOSTIC_H_
#define FIELDDIAGNOSTIC_H_

#include <AMReX_MultiFab.H>

enum struct DiagType{xyz, xz, yz};

class FieldDiagnostic
{
public:
    explicit FieldDiagnostic (int nlev);
    void AllocData (int lev, const amrex::BoxArray& ba, int nfields, const amrex::DistributionMapping& dm);
private:
    /** Vector over levels, all fields */
    amrex::Vector<amrex::MultiFab> m_F;
    DiagType m_diag_type;
    int m_slice_dir;
    int m_nfields;
};

#endif // FIELDDIAGNOSTIC_H_
