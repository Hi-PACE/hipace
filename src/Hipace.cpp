#include "Hipace.H"
#include "particles/deposition/BeamDepositCurrent.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 1000;
}
#endif

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
    pph.query("grid_size_z", m_grid_size_z);
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
    NotifyFinish();
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
    int lev, amrex::Real /*time*/, const amrex::BoxArray& ba, const amrex::DistributionMapping&)
{
    AMREX_ALWAYS_ASSERT(lev == 0);

    // We are going to ignore the DistributionMapping argument and build our own.
    amrex::DistributionMapping dm;
    {
        const amrex::IntVect ncells_global = Geom(0).Domain().length();
        const amrex::IntVect box_size = ba[0].length();  // Uniform box size
        const int nboxes_x = m_numprocs_x;
        const int nboxes_y = m_numprocs_y;
        const int nboxes_z = ncells_global[2] / box_size[2];
        AMREX_ALWAYS_ASSERT(static_cast<long>(nboxes_x) *
                            static_cast<long>(nboxes_y) *
                            static_cast<long>(nboxes_z) == ba.size());
        amrex::Vector<int> procmap;
        // Warning! If we need to do load balancing, we need to update this!
        const int nboxes_x_local = 1;
        const int nboxes_y_local = 1;
        const int nboxes_z_local = nboxes_z / m_numprocs_z;
        for (int k = 0; k < nboxes_z; ++k) {
            int rz = k/nboxes_z_local;
            for (int j = 0; j < nboxes_y; ++j) {
                int ry = j / nboxes_y_local;
                for (int i = 0; i < nboxes_x; ++i) {
                    int rx = i / nboxes_x_local;
                    procmap.push_back(rx+ry*m_numprocs_x+rz*(m_numprocs_x*m_numprocs_y));
                }
            }
        }
        dm.define(std::move(procmap));
    }
    SetDistributionMap(lev, dm); // Let AmrCore know

    m_fields.AllocData(lev, ba, dm);
    m_poisson_solver = FFTPoissonSolver(ba, dm, geom[lev]);
}

void
Hipace::PostProcessBaseGrids (amrex::BoxArray& ba0) const
{
    // This is called by AmrCore::InitFromScratch.
    // The BoxArray made by AmrCore is not what we want.  We will replace it with our own.
    const amrex::IntVect ncells_global = Geom(0).Domain().length();
    amrex::IntVect box_size{ncells_global[0] / m_numprocs_x,
                            ncells_global[1] / m_numprocs_y,
                            m_grid_size_z};
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_size[0]*m_numprocs_x == ncells_global[0],
                                     "# of cells in x-direction is not divisible by hipace.numprocs_x");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_size[1]*m_numprocs_y == ncells_global[1],
                                     "# of cells in y-direction is not divisible by hipace.numprocs_y");

    if (box_size[2] == 0) {
        box_size[2] = ncells_global[2] / m_numprocs_z;
    }

    const int nboxes_x = m_numprocs_x;
    const int nboxes_y = m_numprocs_y;
    const int nboxes_z = ncells_global[2] / box_size[2];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_size[2]*nboxes_z == ncells_global[2],
                                     "# of cells in z-direction is not divisible by # of boxes");

    amrex::BoxList bl;
    for (int k = 0; k < nboxes_z; ++k) {
        for (int j = 0; j < nboxes_y; ++j) {
            for (int i = 0; i < nboxes_x; ++i) {
                amrex::IntVect lo = amrex::IntVect(i,j,k)*box_size;
                amrex::IntVect hi = amrex::IntVect(i+1,j+1,k+1)*box_size - 1;
                bl.push_back(amrex::Box(lo,hi));
            }
        }
    }

    ba0 = amrex::BoxArray(std::move(bl));
}

void
Hipace::Evolve ()
{
    int const lev = 0;
    WriteDiagnostics (0);
    m_fields.getSlices(lev,0).setVal( 0 );
    m_fields.getSlices(lev,1).setVal( 0 );
    m_fields.getSlices(lev,2).setVal( 0 );
    m_fields.getSlices(lev,3).setVal( 0 );

    for (int step = 0; step < m_max_step; ++step)
    {
        Wait();

        //amrex::Print()<<"step "<< step <<"\n";
        /* ---------- Depose current from beam particles ---------- */
        DepositCurrent(m_beam_container, m_fields, geom[lev], lev);

        amrex::MultiFab& fields = m_fields.getF()[lev];
        const amrex::Vector<int> index_array = fields.IndexArray();

        std::cout<<amrex::ParallelDescriptor::MyProc()<<' '<<step<<'\n';
        amrex::Real t1, t2;
        t1 = MPI_Wtime();
        t2 = MPI_Wtime();
        if (amrex::ParallelDescriptor::MyProc() == 1){
            std::cout<<amrex::ParallelDescriptor::MyProc()<< " starts waiting\n";
            while (t2 < t1 + 3){
                t2 = MPI_Wtime();
            }
            std::cout<<amrex::ParallelDescriptor::MyProc()<< " stops  waiting\n";
        }
        
        for (auto it = index_array.rbegin(); it != index_array.rend(); ++it)
        {
            const amrex::Box& bx = fields.box(*it);
            const int islice_hi = bx.bigEnd(Direction::z);
            const int islice_lo = bx.smallEnd(Direction::z);
            for (int islice = islice_hi; islice >= islice_lo; --islice)
            {
                /* ---------- Copy slice islice from m_F to m_slices ---------- */
                m_fields.Copy(lev, islice, FieldCopyType::FtoS, 0, 0, FieldComps::nfields);

                amrex::MultiFab j_slice(m_fields.getSlices(lev, 1), amrex::make_alias,
                                        FieldComps::jx, 3);

                /* xxxxxxxxxx Gather Push Plasma particles transversally xxxxxxxxxx */
                /* xxxxxxxxxx Redistribute Plasma Particles transversally xxxxxxxxxx */
                /* xxxxxxxxxx Deposit current of plasma particles xxxxxxxxxx */

                /* xxxxxxxxxx Transverse FillBoundary current xxxxxxxxxx */
                amrex::ParallelContext::push(m_comm_xy);
                j_slice.FillBoundary(Geom(lev).periodicity());
                amrex::ParallelContext::pop();

                /* ---------- Solve Poisson equation with RHS ---------- */
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

                /* ---------- Transverse FillBoundary By ---------- */
                amrex::ParallelContext::push(m_comm_xy);
                m_fields.getSlices(lev,1).setVal( m_fields.getSlices(lev,2).max(FieldComps::Ez)+1 );
                
                lhs.FillBoundary(Geom(lev).periodicity());
                amrex::ParallelContext::pop();

                /* ---------- Copy back from the slice MultiFab m_slices to the main field m_F ---------- */
                m_fields.Copy(lev, islice, FieldCopyType::StoF, 0, 0, FieldComps::nfields);

                /* ---------- Shift slices ---------- */
                m_fields.ShiftSlices(lev);
            }
        }
        /* xxxxxxxxxx Gather and push beam particles xxxxxxxxxx */
        // Slices have already been shifted, so send
        // slices {2,3} from upstream to {2,3} in downstream.
        Notify();
    }

    WriteDiagnostics (1);
}

void
Hipace::Wait ()
{
#ifdef AMREX_USE_MPI
    if (m_rank_z != m_numprocs_z-1) {
        const int lev = 0;
        amrex::MultiFab& slice2 = m_fields.getSlices(lev, 2);
        amrex::MultiFab& slice3 = m_fields.getSlices(lev, 3);
        // Note that there is only one local Box in slice multifab's boxarray.
        const int box_index = slice2.IndexArray()[0];
        amrex::Array4<amrex::Real> const& slice_fab2 = slice2.array(box_index);
        amrex::Array4<amrex::Real> const& slice_fab3 = slice3.array(box_index);
        const amrex::Box& bx = slice2.boxArray()[box_index]; // does not include ghost cells
        const std::size_t nreals_valid_slice2 = bx.numPts()*slice_fab2.nComp();
        const std::size_t nreals_valid_slice3 = bx.numPts()*slice_fab3.nComp();
        const std::size_t nreals_total = nreals_valid_slice2 + nreals_valid_slice3;
        auto recv_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real)*nreals_total);
        auto const buf2 = amrex::makeArray4(recv_buffer,
                                            bx, slice_fab2.nComp());
        auto const buf3 = amrex::makeArray4(recv_buffer+nreals_valid_slice2,
                                            bx, slice_fab3.nComp());
        MPI_Status status;
        MPI_Recv(recv_buffer, nreals_total,
                 amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 m_rank_z+1, comm_z_tag, m_comm_z, &status);
        amrex::ParallelFor
            (bx, slice_fab2.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 slice_fab2(i,j,k,n) = buf2(i,j,k,n);
             },
             bx, slice_fab3.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 slice_fab3(i,j,k,n) = buf3(i,j,k,n);
             });
        amrex::The_Pinned_Arena()->free(recv_buffer);
    }
#endif
}

void
Hipace::Notify ()
{
    // Send from slices 2 and 3 (or main MultiFab's first two valid slabs) to receiver's slices 2
    // and 3.
#ifdef AMREX_USE_MPI
    if (m_rank_z != 0) {
        NotifyFinish(); // finish the previous send

        const int lev = 0;
        const amrex::MultiFab& slice2 = m_fields.getSlices(lev, 2);
        const amrex::MultiFab& slice3 = m_fields.getSlices(lev, 3);
        // Note that there is only one local Box in slice multifab's boxarray.
        const int box_index = slice2.IndexArray()[0];
        amrex::Array4<amrex::Real const> const& slice_fab2 = slice2.array(box_index);
        amrex::Array4<amrex::Real const> const& slice_fab3 = slice3.array(box_index);
        const amrex::Box& bx = slice2.boxArray()[box_index]; // does not include ghost cells
        const std::size_t nreals_valid_slice2 = bx.numPts()*slice_fab2.nComp();
        const std::size_t nreals_valid_slice3 = bx.numPts()*slice_fab3.nComp();
        const std::size_t nreals_total = nreals_valid_slice2 + nreals_valid_slice3;
        m_send_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real)*nreals_total);
        auto const buf2 = amrex::makeArray4(m_send_buffer,
                                            bx, slice_fab2.nComp());
        auto const buf3 = amrex::makeArray4(m_send_buffer+nreals_valid_slice2,
                                            bx, slice_fab3.nComp());
        amrex::ParallelFor
            (bx, slice_fab2.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 buf2(i,j,k,n) = slice_fab2(i,j,k,n);
             },
             bx, slice_fab3.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 buf3(i,j,k,n) = slice_fab3(i,j,k,n);
             });
        MPI_Isend(m_send_buffer, nreals_total,
                  amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                  m_rank_z-1, comm_z_tag, m_comm_z, &m_send_request);
    }
#endif
}

void
Hipace::NotifyFinish ()
{
#ifdef AMREX_USE_MPI
    if (m_rank_z != 0) {
        if (m_send_buffer) {
            MPI_Status status;
            MPI_Wait(&m_send_request, &status);
            amrex::The_Pinned_Arena()->free(m_send_buffer);
            m_send_buffer = nullptr;
        }
    }
#endif
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
