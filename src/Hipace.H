#ifndef HIPACE_H_
#define HIPACE_H_

#include "fields/Fields.H"
#include "fields/fft_poisson_solver/FFTPoissonSolver.H"
#include "particles/PlasmaParticleContainer.H"
#include "particles/BeamParticleContainer.H"
#include "Constants.H"

#include <AMReX_AmrCore.H>

class Hipace final : public amrex::AmrCore
{
public:
    Hipace ();

    ~Hipace ();

    static Hipace& GetInstance ();

    // Virtual functions need to be defined for pure virtual class AmrCore
    void MakeNewLevelFromScratch (
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    void ErrorEst (
                   int /*lev*/, amrex::TagBoxArray& /*tags*/, amrex::Real /*time*/, int /*ngrow*/) override {}

    void MakeNewLevelFromCoarse (
                                 int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                                 const amrex::DistributionMapping& /*dm*/) override {}

    void RemakeLevel (
                      int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                      const amrex::DistributionMapping& /*dm*/) override {}

    void ClearLevel (int /*lev*/) override {}

    void PostProcessBaseGrids (amrex::BoxArray& ba0) const override;

    void InitData ();

    void Evolve ();

    void Wait ();
    void Notify ();
    void NotifyFinish ();

    // Is rank in the same transverse communicator w/ me?
    bool InSameTransverseCommunicator (int rank) const;

    void WriteDiagnostics (int step);

    PhysConst get_phys_const () {return m_phys_const;};

    MPI_Comm m_comm_xy = MPI_COMM_NULL;
    MPI_Comm m_comm_z = MPI_COMM_NULL;
    int m_numprocs_x = 1;
    int m_numprocs_y = 1;
    int m_numprocs_z = 0;
    int m_rank_xy = 0;
    int m_rank_z = 0;
    int m_grid_size_z = 0;
    amrex::Real* m_send_buffer = nullptr;
    MPI_Request m_send_request = MPI_REQUEST_NULL;

    Fields m_fields;
    FFTPoissonSolver m_poisson_solver;
    BeamParticleContainer m_beam_container;
    PlasmaParticleContainer m_plasma_container;
    int m_max_step = 0;

    static bool m_normalized_units;
    PhysConst m_phys_const;
    static int m_depos_order_xy;
    static int m_depos_order_z;

private:
    static Hipace* m_instance;
};

#endif
