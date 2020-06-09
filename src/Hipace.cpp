#include "Hipace.H"
#include "particles/deposition/BeamDepositCurrent.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

Hipace::Hipace () :
    m_fields(this),
    m_beam_container(this),
    m_plasma_container(this)
{
    amrex::ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
    pp.query("max_step", m_max_step);

    amrex::ParmParse pph("hipace");
    pph.query("numprocs_x", m_numprocs_x);
    pph.query("numprocs_y", m_numprocs_y);
    m_numprocs_z = amrex::ParallelDescriptor::NProcs() / (m_numprocs_x*m_numprocs_y);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_x*m_numprocs_y*m_numprocs_z
                                     == amrex::ParallelDescriptor::NProcs(),
                                     "Check hipace.numprocs_x and hipace.numprocs_y");
#ifdef AMREX_USE_MPI
    int myproc = amrex::ParallelDescriptor::MyProc();
    m_rank_z = myproc/(m_numprocs_x*m_numprocs_y);
    MPI_Comm_split(amrex::ParallelDescriptor::Communicator(), m_rank_z, myproc, &m_comm_xy);
    MPI_Comm_rank(m_comm_xy, &m_rank_xy);
    MPI_Comm_split(amrex::ParallelDescriptor::Communicator(), m_rank_xy, myproc, &m_comm_z);
#endif
}

Hipace::~Hipace ()
{
#ifdef AMREX_USE_MPI
    MPI_Comm_free(&m_comm_xy);
    MPI_Comm_free(&m_comm_z);
#endif
}

void
Hipace::InitData ()
{
    amrex::Vector<amrex::IntVect> new_max_grid_size;
    for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
        amrex::IntVect mgs = maxGridSize(ilev);
        mgs[0] = mgs[1] = 1024000000; // disable domain decomposition in x and y directions
        new_max_grid_size.push_back(mgs);
    }
    SetMaxGridSize(new_max_grid_size);

    AmrCore::InitFromScratch(0.0); // function argument is time
    m_beam_container.InitData(geom[0]);
    m_plasma_container.InitData(geom[0]);
}

void
Hipace::MakeNewLevelFromScratch (
    int lev, amrex::Real /*time*/, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    AMREX_ALWAYS_ASSERT(lev == 0);
    m_fields.AllocData(lev, ba, dm);
    m_poisson_solver = FFTPoissonSolver(ba, dm, geom[lev]);
}

void
Hipace::Evolve ()
{
    int const lev = 0;
    WriteDiagnostics (0);
    for (int step = 0; step < m_max_step; ++step)
    {
        amrex::Print()<<"step "<< step <<"\n";
        DepositCurrent(m_beam_container, m_fields, geom[lev], lev);
        for ( amrex::MFIter mfi(m_fields.getF()[lev], false); mfi.isValid(); ++mfi ){
            const amrex::Box& bx = mfi.tilebox();
            const int nslices = bx.hiVect()[Direction::z]+1;
            for (int islice=nslices-1; islice>=0; islice--){
                amrex::Print()<<islice<<'\n';
                // Copy slice islice from m_F to m_slices
                m_fields.Copy(lev, islice, FieldCopyType::FtoS, 0, 0, FieldComps::nfields);
                // Left-Hand Side for Poisson equation is By in the slice MF
                amrex::MultiFab lhs(m_fields.getSlices(lev, 1), amrex::make_alias,
                                    FieldComps::By, 1);
                // Left-Hand Side for Poisson equation: allocate a tmp MultiFab
                amrex::MultiFab rhs = amrex::MultiFab(m_fields.getSlices(lev, 1).boxArray(),
                                                      m_fields.getSlices(lev, 1).distributionMap,
                                                      1, 0);
                // Left-Hand Side for Poisson equation: compute d_x(jz) from the slice MF,
                // and store in tmp MultiFab rhs
                m_fields.TransverseDerivative(m_fields.getSlices(lev, 1), rhs, Direction::x,
                                              geom[0].CellSize(Direction::x), FieldComps::jz);
                rhs.mult(PhysConst::mu0);
                // Solve Poisson equation, the result (lhs) is in the slice MultiFab m_slices
                m_poisson_solver.SolvePoissonEquation(rhs, lhs);
                // Copy back from the slice MultiFab m_slices to the main field m_F
                m_fields.Copy(lev, islice, FieldCopyType::StoF, 0, 0, FieldComps::nfields);
            }
        }
    }
    WriteDiagnostics (1);
}

void
Hipace::WriteDiagnostics (int step)
{
    // Write fields
    const std::string filename = amrex::Concatenate("plt", step);
    const int nlev = 1;
    const amrex::Vector< std::string > varnames {"ExmBy", "EypBx", "Ez", "Bx", "By", "Bz",
                                                 "jx", "jy", "jz"};
    const int time = 0.;
    const amrex::IntVect local_ref_ratio {1, 1, 1};
    amrex::Vector<std::string> rfs;
    amrex::WriteMultiLevelPlotfile(filename, nlev,
                                   amrex::GetVecOfConstPtrs(m_fields.getF()),
                                   varnames, Geom(),
                                   time, {step}, {local_ref_ratio},
                                   "HyperCLaw-V1.1",
                                   "Level_",
                                   "Cell",
                                   rfs
        );

    // Write beam particles
    {
        amrex::Vector<int> plot_flags(BeamIdx::nattribs, 1);
        amrex::Vector<int> int_flags(BeamIdx::nattribs, 1);
        amrex::Vector<std::string> real_names {"w","ux","uy","uz"};
        AMREX_ALWAYS_ASSERT(real_names.size() == BeamIdx::nattribs);
        amrex::Vector<std::string> int_names {};
        m_beam_container.WritePlotFile(
            filename, "beam",
            plot_flags, int_flags,
            real_names, int_names);
    }

    // Write plasma particles
    {
        amrex::Vector<int> plot_flags(PlasmaIdx::nattribs, 1);
        amrex::Vector<int> int_flags(PlasmaIdx::nattribs, 1);
        amrex::Vector<std::string> real_names {
            "w","ux","uy", "phi",
            "Fx1", "Fx2", "Fx3", "Fx4", "Fx5",
            "Fy1", "Fy2", "Fy3", "Fy4", "Fy5",
            "Fux1", "Fux2", "Fux3", "Fux4", "Fux5",
            "Fuy1", "Fuy2", "Fuy3", "Fuy4", "Fuy5",
            "Fphi1", "Fphi2", "Fphi3", "Fphi4", "Fphi5",
        };
        AMREX_ALWAYS_ASSERT(real_names.size() == PlasmaIdx::nattribs);
        amrex::Vector<std::string> int_names {};
        m_plasma_container.WritePlotFile(
            filename, "plasma",
            plot_flags, int_flags,
            real_names, int_names);
    }
}
