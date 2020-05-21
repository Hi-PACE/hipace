#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>

class Fields
{
public:
    Fields () = default;
    void InitData () {};
private:
    amrex::MultiFab m_F;
};

#endif
