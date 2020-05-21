#ifndef HIPACE_H_
#define HIPACE_H_

#include "Fields/Fields.H"

#include <AMReX_AmrCore.H>

class Hipace final : public amrex::AmrCore
{
public:
    Hipace () = default;

    // Virtual functions need to be defined for pure virtual class AmrCore
    void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
    void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override {};
    void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override {};
    void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override {};



    void ClearLevel (int lev) override {};
    void InitData ();
    Fields m_fields;

    //MultiBeamContainer beam_containers;
    //MultiPlasmaContainer plasma_containers;
};

#endif
