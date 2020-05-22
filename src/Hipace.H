#ifndef HIPACE_H_
#define HIPACE_H_

#include "fields/Fields.H"
#include "particles/MultiPlasmaParticleContainer.H"
#include "particles/MultiBeamParticleContainer.H"

#include <AMReX_AmrCore.H>

class Hipace final : public amrex::AmrCore
{
public:
    Hipace ();

    // Virtual functions need to be defined for pure virtual class AmrCore
    void MakeNewLevelFromScratch (
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;
    void ErrorEst (
        int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override {};
    void MakeNewLevelFromCoarse (
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override {};
    void RemakeLevel (
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override {};
    void ClearLevel (int lev) override {};
    
    void InitData ();
    void Evolve ();
    void WriteDiagnostics (int step);

    Fields m_fields;
    MultiBeamParticleContainer m_beam_containers;
    MultiPlasmaParticleContainer m_plasma_containers;
    int m_max_step;
};

#endif
