#include "Hipace.H"
#include "particles/deposition/BeamDepositCurrent.H"
#include "particles/deposition/PlasmaDepositCurrent.H"
#include "particles/pusher/PlasmaParticleAdvance.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 1000;
}
#endif

Hipace* Hipace::m_instance = nullptr;

bool Hipace::m_normalized_units = false;
int Hipace::m_depos_order_xy = 2;
int Hipace::m_depos_order_z = 0;

Hipace&
Hipace::GetInstance ()
{
    if (!m_instance) {
        m_instance = new Hipace();
    }
    return *m_instance;
}

Hipace::Hipace () :
    m_fields(this),
    m_beam_container(this),
    m_plasma_container(this)
{
    m_instance = this;

    amrex::ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
    pp.query("max_step", m_max_step);

    amrex::ParmParse pph("hipace");
    pph.query("normalized_units", m_normalized_units);
    if (m_normalized_units){
        m_phys_const = make_constants_normalized();
    } else {
        m_phys_const = make_constants_SI();
    }
    pph.query("numprocs_x", m_numprocs_x);
    pph.query("numprocs_y", m_numprocs_y);
    pph.query("grid_size_z", m_grid_size_z);
    pph.query("depos_order_xy", m_depos_order_xy);
    pph.query("depos_order_z", m_depos_order_z);
    pph.query("do_plot", m_do_plot);
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

bool
Hipace::InSameTransverseCommunicator (int rank) const
{
    return rank/(m_numprocs_x*m_numprocs_y) == m_rank_z;
}

void
Hipace::InitData ()
{
    BL_PROFILE("Hipace::InitData()");
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
    // The Poisson solver operates on transverse slices only.
    // The constructor takes the BoxArray and the DistributionMap of a slice,
    // so the FFTPlans are built on a slice.
    m_poisson_solver = FFTPoissonSolver(
        m_fields.getSlices(lev, 1).boxArray(),
        m_fields.getSlices(lev, 1).DistributionMap(),
        geom[lev]);
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
    BL_PROFILE("Hipace::Evolve()");
    int const lev = 0;
    if (m_do_plot) WriteDiagnostics(0);
    for (int step = 0; step < m_max_step; ++step)
    {
        Wait();

        amrex::Print()<<"step "<< step <<"\n";

        /* ---------- Depose current from beam particles ---------- */
        m_fields.getF(lev).setVal(0.);
        DepositCurrent(m_beam_container, m_fields, geom[lev], lev);

        amrex::MultiFab& fields = m_fields.getF()[lev];
        const amrex::Vector<int> index_array = fields.IndexArray();
        for (auto it = index_array.rbegin(); it != index_array.rend(); ++it)
        {
            const amrex::Box& bx = fields.box(*it);
            const int islice_hi = bx.bigEnd(Direction::z);
            const int islice_lo = bx.smallEnd(Direction::z);
            for (int islice = islice_hi; islice >= islice_lo; --islice)
            {
                m_fields.Copy(lev, islice, FieldCopyType::FtoS, 0, 0, FieldComps::nfields);

                AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                                       ToSlice::This,
                                       true, false, false, lev);

                amrex::ParallelContext::push(m_comm_xy);
                m_plasma_container.Redistribute();
                amrex::ParallelContext::pop();

                DepositCurrent(m_plasma_container, m_fields, ToSlice::This,
                               geom[lev], lev);

                amrex::ParallelContext::push(m_comm_xy);
                // need to exchange jx jy jz rho
                amrex::MultiFab j_slice(m_fields.getSlices(lev, 1),
                                         amrex::make_alias, FieldComps::jx, 4);
                j_slice.SumBoundary(Geom(lev).periodicity());
                amrex::ParallelContext::pop();

                SolvePoissonEz(lev);
                SolvePoissonExmByAndEypBx(lev);
                SolvePoissonBz(lev);

                PredictorCorrectorLoopToSolveBxBy(lev);

                /* ------ Copy slice from m_slices to the main field m_F ------ */
                m_fields.Copy(lev, islice, FieldCopyType::StoF, 0, 0, FieldComps::nfields);

                m_fields.ShiftSlices(lev);
            }
        }
        /* xxxxxxxxxx Gather and push beam particles xxxxxxxxxx */

        // Slices have already been shifted, so send
        // slices {2,3} from upstream to {2,3} in downstream.
        Notify();
    }

    if (m_do_plot) WriteDiagnostics(1);
}

void Hipace::SolvePoissonExmByAndEypBx (const int lev)
{
    /* Solves Laplacian(-Psi) =  1/episilon0 * (rho-Jz/c) and
     * calculates Ex-c By, Ey + c Bx from  grad(-Psi)
     */
    BL_PROFILE("Hipace::SolveExmByAndEypBx()");
    // Left-Hand Side for Poisson equation is Psi in the slice MF
    amrex::MultiFab lhs(m_fields.getSlices(lev, 1), amrex::make_alias,
                        FieldComps::Psi, 1);

    // calculating the right-hand side 1/episilon0 * (rho-Jz/c)
    amrex::MultiFab::Copy(m_poisson_solver.StagingArea(), m_fields.getSlices(lev, 1),
                              FieldComps::jz, 0, 1, 0);
    m_poisson_solver.StagingArea().mult(-1./m_phys_const.c);
    amrex::MultiFab::Add(m_poisson_solver.StagingArea(), m_fields.getSlices(lev, 1),
                          FieldComps::rho, 0, 1, 0);


    m_poisson_solver.SolvePoissonEquation(lhs);
    /* ---------- Transverse FillBoundary Psi ---------- */
    amrex::ParallelContext::push(m_comm_xy);
    lhs.FillBoundary(Geom(lev).periodicity());
    amrex::ParallelContext::pop();

    /* Compute ExmBy and Eypbx from grad(-psi) */
    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_fields.getSlices(lev, 1),
        Direction::x,
        geom[0].CellSize(Direction::x),
        1.,
        SliceOperatorType::Assign,
        FieldComps::Psi,
        FieldComps::ExmBy);

    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_fields.getSlices(lev, 1),
        Direction::y,
        geom[0].CellSize(Direction::y),
        1.,
        SliceOperatorType::Assign,
        FieldComps::Psi,
        FieldComps::EypBx);
}

void Hipace::SolvePoissonEz (const int lev)
{
    /* Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy)) */
    BL_PROFILE("Hipace::SolvePoissonEz()");
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(m_fields.getSlices(lev, 1), amrex::make_alias,
                        FieldComps::Ez, 1);
    // Right-Hand Side for Poisson equation: compute 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
    // from the slice MF, and store in the staging area of m_poisson_solver
    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_poisson_solver.StagingArea(),
        Direction::x,
        geom[0].CellSize(Direction::x),
        1./(m_phys_const.ep0*m_phys_const.c),
        SliceOperatorType::Assign,
        FieldComps::jx);

    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_poisson_solver.StagingArea(),
        Direction::y,
        geom[0].CellSize(Direction::y),
        1./(m_phys_const.ep0*m_phys_const.c),
        SliceOperatorType::Add,
        FieldComps::jy);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(lhs);
}

void Hipace::SolvePoissonBx (amrex::MultiFab& Bx_iter, const int lev)
{
    /* Solves Laplacian(Bx) = mu_0*(- d_y(jz) + d_z(jy) ) */
    BL_PROFILE("Hipace::SolvePoissonBx()");

    // Right-Hand Side for Poisson equation: compute -mu_0*d_y(jz) from the slice MF,
    // and store in the staging area of m_poisson_solver
    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_poisson_solver.StagingArea(),
        Direction::y,
        geom[0].CellSize(Direction::y),
        -m_phys_const.mu0,
        SliceOperatorType::Assign,
        FieldComps::jz);

    m_fields.LongitudinalDerivative(
        m_fields.getSlices(lev, 2),
        m_fields.getSlices(lev, 0),
        m_poisson_solver.StagingArea(),
        geom[0].CellSize(Direction::z),
        m_phys_const.mu0,
        SliceOperatorType::Add,
        FieldComps::jy, FieldComps::jy);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(Bx_iter);
}

void Hipace::SolvePoissonBy (amrex::MultiFab& By_iter, const int lev)
{
    /* Solves Laplacian(By) = mu_0*(d_x(jz) - d_z(jx) ) */
    BL_PROFILE("Hipace::SolvePoissonBy()");

    // Right-Hand Side for Poisson equation: compute mu_0*d_x(jz) from the slice MF,
    // and store in the staging area of m_poisson_solver
    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_poisson_solver.StagingArea(),
        Direction::x,
        geom[0].CellSize(Direction::x),
        m_phys_const.mu0,
        SliceOperatorType::Assign,
        FieldComps::jz);

    m_fields.LongitudinalDerivative(
        m_fields.getSlices(lev, 2),
        m_fields.getSlices(lev, 0),
        m_poisson_solver.StagingArea(),
        geom[0].CellSize(Direction::z),
        -m_phys_const.mu0,
        SliceOperatorType::Add,
        FieldComps::jx, FieldComps::jx);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(By_iter);
}

void Hipace::SolvePoissonBz (const int lev)
{
    /* Solves Laplacian(Bz) = mu_0*(d_y(jx) - d_x(jy)) */
    BL_PROFILE("Hipace::SolvePoissonBz()");
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(m_fields.getSlices(lev, 1), amrex::make_alias,
                        FieldComps::Bz, 1);
    // Right-Hand Side for Poisson equation: compute mu_0*(d_y(jx) - d_x(jy))
    // from the slice MF, and store in the staging area of m_poisson_solver
    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_poisson_solver.StagingArea(),
        Direction::y,
        geom[0].CellSize(Direction::y),
        m_phys_const.mu0,
        SliceOperatorType::Assign,
        FieldComps::jx);

    m_fields.TransverseDerivative(
        m_fields.getSlices(lev, 1),
        m_poisson_solver.StagingArea(),
        Direction::x,
        geom[0].CellSize(Direction::x),
        -m_phys_const.mu0,
        SliceOperatorType::Add,
        FieldComps::jy);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(lhs);
}

void Hipace::InitialBfieldGuess (const int lev)
{
    /* Sets the initial guess of the B field from the two previous slices
     */
    BL_PROFILE("Hipace::InitialBfieldGuess()");

    const double factor = 1.;
    /* later, the factor should be:
     * exp(-0.5 * pow(rel_avg_Bdiff / ( 2.5 * fld_predcorr_tol_b ), 2));
     */
    amrex::MultiFab::LinComb(m_fields.getSlices(lev, 1),
                             1+factor,
                             m_fields.getSlices(lev, 2),
                             FieldComps::Bx,
                             -factor,
                             m_fields.getSlices(lev, 3),
                             FieldComps::Bx,
                             FieldComps::Bx,
                             1,
                             0);

     amrex::MultiFab::LinComb(m_fields.getSlices(lev, 1),
                             1+factor,
                             m_fields.getSlices(lev, 2),
                             FieldComps::By,
                             -factor,
                             m_fields.getSlices(lev, 3),
                             FieldComps::By,
                             FieldComps::By,
                             1,
                             0);

}

void Hipace::MixAndShiftBfields (amrex::MultiFab& B_iter, amrex::MultiFab& B_tmp,
                                 const int field_comp, const int lev)
{
    /* Mixes the B field according to B = a*B + (1-a)*( c*B_iter + d*B_prev_iter),
     * with a,c,d mixing coefficients.
     */
    BL_PROFILE("Hipace::MixAndShiftBfields()");

    /* mixing factor between Bx and the mixed iterated B field */
    const double mixing_factor = 0.1;
    //later, the factor should be defined in the input deck (goal, have this one flexible)

    /* Mixing factors to mix the current and previous iteration of the B field */
    const double c_B_iter = 0.5;
    const double c_B_prev_iter = 0.5;
    /* later, this should be
     *   c_B_iter = rel_avg_Bdiff_iter_m_1 / ( rel_avg_Bdiff + rel_avg_Bdiff_iter_m_1 );
     *   c_B_prev_iter = rel_avg_Bdiff / ( rel_avg_Bdiff + rel_avg_Bdiff_iter_m_1 );
     */

    /* calculating the mixed temporary B field  B_tmp = c*B_iter + d*B_prev_iter*/
    amrex::MultiFab::LinComb(B_tmp, c_B_iter, B_iter, 0, c_B_prev_iter,
                             B_tmp, 0, 0, 1, 0);

    /* calculating the mixed B field  B = a*B + (1-a)*B_tmp */
    amrex::MultiFab::LinComb(m_fields.getSlices(lev, 1), 1-mixing_factor,
                             m_fields.getSlices(lev, 1), field_comp,
                             mixing_factor, B_tmp, 0, field_comp, 1, 0);

    /* Shifting the B field from the current iteration to the previous iteration */
    amrex::MultiFab::Copy(B_tmp, B_iter, 0, 0, 1, 0);


}

void Hipace::PredictorCorrectorLoopToSolveBxBy (const int lev)
{
    /* Guess Bx and By */
    InitialBfieldGuess(lev);
    amrex::ParallelContext::push(m_comm_xy);
     // exchange ExmBy EypBx Ez Bx By Bz
    m_fields.getSlices(lev, 1).FillBoundary(Geom(lev).periodicity());
    amrex::ParallelContext::pop();

    /* creating temporary Bx and By arrays for the current and previous iteration */
    amrex::MultiFab Bx_iter(m_fields.getSlices(lev, 1).boxArray(),
                            m_fields.getSlices(lev, 1).DistributionMap(), 1,
                            m_fields.getSlices(lev, 1).nGrowVect());
    amrex::MultiFab By_iter(m_fields.getSlices(lev, 1).boxArray(),
                            m_fields.getSlices(lev, 1).DistributionMap(), 1,
                            m_fields.getSlices(lev, 1).nGrowVect());
    amrex::MultiFab Bx_prev_iter(m_fields.getSlices(lev, 1).boxArray(),
                                 m_fields.getSlices(lev, 1).DistributionMap(), 1,
                                 m_fields.getSlices(lev, 1).nGrowVect());
    amrex::MultiFab::Copy(Bx_prev_iter, m_fields.getSlices(lev, 1),
                          FieldComps::Bx, 0, 1, 0);
    amrex::MultiFab By_prev_iter(m_fields.getSlices(lev, 1).boxArray(),
                                 m_fields.getSlices(lev, 1).DistributionMap(), 1,
                                 m_fields.getSlices(lev, 1).nGrowVect());
    amrex::MultiFab::Copy(By_prev_iter, m_fields.getSlices(lev, 1),
                          FieldComps::By, 0, 1, 0);

    /* creating aliases to the current in the next slice.
     * This needs to be reset after each push to the next slice */
    amrex::MultiFab jx_next(m_fields.getSlices(lev, 0), amrex::make_alias,
                            FieldComps::jx, 1);
    amrex::MultiFab jy_next(m_fields.getSlices(lev, 0), amrex::make_alias,
                            FieldComps::jy, 1);


    /* shift force terms, update force terms using guessed Bx and By */
    AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                           ToSlice::This,
                           false, true, true, lev);

    /* Begin of predictor corrector loop  */
    for (int pred_number=0; pred_number<20; pred_number++)
    {
        /* Push particles to the next slice */
        AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                               ToSlice::Next,
                               true, false, false, lev);

        /* deposit current to next slice */
        DepositCurrent(m_plasma_container, m_fields, ToSlice::Next,
                       geom[lev], lev);
        amrex::ParallelContext::push(m_comm_xy);
        // need to exchange jx jy jz rho
        amrex::MultiFab j_slice_next(m_fields.getSlices(lev, 0),
                                     amrex::make_alias, FieldComps::jx, 4);
        j_slice_next.SumBoundary(Geom(lev).periodicity());
        amrex::ParallelContext::pop();

        /* Calculate Bx and By */
        SolvePoissonBx(Bx_iter, lev);
        SolvePoissonBy(By_iter, lev);

        /* Mixing the calculated B fields to the actual B field and shifting iterated B fields */
        MixAndShiftBfields(Bx_iter, Bx_prev_iter, FieldComps::Bx, lev);
        MixAndShiftBfields(By_iter, By_prev_iter, FieldComps::By, lev);

        /* resetting current in the next slice to clean temporarily used current*/
        jx_next.setVal(0.);
        jy_next.setVal(0.);

        amrex::ParallelContext::push(m_comm_xy);
         // exchange Bx By
        m_fields.getSlices(lev, 1).FillBoundary(Geom(lev).periodicity());
        amrex::ParallelContext::pop();

        /* Update force terms using the calculated Bx and By */
        AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                               ToSlice::Next,
                               false, true, false, lev);
    } /* end of predictor corrector loop */
}

void
Hipace::Wait ()
{
    BL_PROFILE("Hipace::Wait()");
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
    BL_PROFILE("Hipace::Notify()");
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
    BL_PROFILE("Hipace::WriteDiagnostics()");
    // Write fields
    const std::string filename = amrex::Concatenate("plt", step);
    const int nlev = 1;
    const amrex::Vector< std::string > varnames {"ExmBy", "EypBx", "Ez", "Bx", "By", "Bz",
                                                 "jx", "jy", "jz", "rho", "Psi"};
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
            "w","ux","uy", "psi",
            "x_temp", "y_temp", "w_temp", "ux_temp", "uy_temp", "psi_temp",
            "Fx1", "Fx2", "Fx3", "Fx4", "Fx5",
            "Fy1", "Fy2", "Fy3", "Fy4", "Fy5",
            "Fux1", "Fux2", "Fux3", "Fux4", "Fux5",
            "Fuy1", "Fuy2", "Fuy3", "Fuy4", "Fuy5",
            "Fpsi1", "Fpsi2", "Fpsi3", "Fpsi4", "Fpsi5",
        };
        AMREX_ALWAYS_ASSERT(real_names.size() == PlasmaIdx::nattribs);
        amrex::Vector<std::string> int_names {};
        m_plasma_container.WritePlotFile(
            filename, "plasma",
            plot_flags, int_flags,
            real_names, int_names);
    }
}
