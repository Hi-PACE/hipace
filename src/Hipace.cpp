#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "particles/BinSort.H"
#include "particles/BoxSort.H"
#include "utils/IOUtil.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>

#include <algorithm>
#include <memory>

#ifdef AMREX_USE_MPI
namespace {
    constexpr int ncomm_z_tag = 1001;
    constexpr int pcomm_z_tag = 1002;
}
#endif

Hipace* Hipace::m_instance = nullptr;

int Hipace::m_max_step = 0;
amrex::Real Hipace::m_dt = 0.0;
bool Hipace::m_normalized_units = false;
int Hipace::m_verbose = 0;
int Hipace::m_depos_order_xy = 2;
int Hipace::m_depos_order_z = 0;
amrex::Real Hipace::m_predcorr_B_error_tolerance = 4e-2;
int Hipace::m_predcorr_max_iterations = 30;
amrex::Real Hipace::m_predcorr_B_mixing_factor = 0.05;
bool Hipace::m_do_beam_jx_jy_deposition = true;
bool Hipace::m_do_device_synchronize = false;
int Hipace::m_beam_injection_cr = 1;
amrex::Real Hipace::m_external_ExmBy_slope = 0.;
amrex::Real Hipace::m_external_Ez_slope = 0.;
amrex::Real Hipace::m_external_Ez_uniform = 0.;

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
    m_multi_beam(this),
    m_multi_plasma(this)
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
    pph.query("dt", m_dt);
    pph.query("verbose", m_verbose);
    pph.query("numprocs_x", m_numprocs_x);
    pph.query("numprocs_y", m_numprocs_y);
    pph.query("grid_size_z", m_grid_size_z);
    pph.query("depos_order_xy", m_depos_order_xy);
    pph.query("depos_order_z", m_depos_order_z);
    pph.query("predcorr_B_error_tolerance", m_predcorr_B_error_tolerance);
    pph.query("predcorr_max_iterations", m_predcorr_max_iterations);
    pph.query("predcorr_B_mixing_factor", m_predcorr_B_mixing_factor);
    pph.query("output_period", m_output_period);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_output_period != 0,
                                     "To avoid output, please use output_period = -1.");
    pph.query("beam_injection_cr", m_beam_injection_cr);
    m_numprocs_z = amrex::ParallelDescriptor::NProcs() / (m_numprocs_x*m_numprocs_y);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_z <= m_max_step,
                                     "Please use more or equal time steps than number of ranks");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_x*m_numprocs_y*m_numprocs_z
                                     == amrex::ParallelDescriptor::NProcs(),
                                     "Check hipace.numprocs_x and hipace.numprocs_y");
    pph.query("do_beam_jx_jy_deposition", m_do_beam_jx_jy_deposition);
    pph.query("do_device_synchronize", m_do_device_synchronize);
    pph.query("external_ExmBy_slope", m_external_ExmBy_slope);
    pph.query("external_Ez_slope", m_external_Ez_slope);
    pph.query("external_Ez_uniform", m_external_Ez_uniform);

#ifdef AMREX_USE_MPI
    pph.query("skip_empty_comms", m_skip_empty_comms);
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
Hipace::DefineSliceGDB (const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    std::map<int,amrex::Vector<amrex::Box> > boxes;
    for (int i = 0; i < ba.size(); ++i) {
        int rank = dm[i];
        if (InSameTransverseCommunicator(rank)) {
            boxes[rank].push_back(ba[i]);
        }
    }

    // We assume each process may have multiple Boxes longitude direction, but only one Box in the
    // transverse direction.  The union of all Boxes on a process is rectangular.  The slice
    // BoxArray therefore has one Box per process.  The Boxes in the slice BoxArray have one cell in
    // the longitude direction.  We will use the lowest longitude index in each process to construct
    // the Boxes.  These Boxes do not have any overlaps. Transversely, there are no gaps between
    // them.

    amrex::BoxList bl;
    amrex::Vector<int> procmap;
    for (auto const& kv : boxes) {
        int const iproc = kv.first;
        auto const& boxes_i = kv.second;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boxes_i.size() > 0,
                                         "We assume each process has at least one Box");
        amrex::Box bx = boxes_i[0];
        for (int j = 1; j < boxes_i.size(); ++j) {
            amrex::Box const& bxj = boxes_i[j];
            for (int idim = 0; idim < Direction::z; ++idim) {
                AMREX_ALWAYS_ASSERT(bxj.smallEnd(idim) == bx.smallEnd(idim));
                AMREX_ALWAYS_ASSERT(bxj.bigEnd(idim) == bx.bigEnd(idim));
                if (bxj.smallEnd(Direction::z) < bx.smallEnd(Direction::z)) {
                    bx = bxj;
                }
            }
        }
        bx.setBig(Direction::z, bx.smallEnd(Direction::z));
        bl.push_back(bx);
        procmap.push_back(iproc);
    }

    // Slice BoxArray
    m_slice_ba = amrex::BoxArray(std::move(bl));

    // Slice DistributionMapping
    m_slice_dm = amrex::DistributionMapping(std::move(procmap));

    // Slice Geometry
    constexpr int lev = 0;
    const int dir = AMREX_SPACEDIM-1;
    // Set the lo and hi of domain and probdomain in the z direction
    amrex::RealBox tmp_probdom = Geom(lev).ProbDomain();
    amrex::Box tmp_dom = Geom(lev).Domain();
    const amrex::Real dx = Geom(lev).CellSize(dir);
    const amrex::Real hi = Geom(lev).ProbHi(dir);
    const amrex::Real lo = hi - dx;
    tmp_probdom.setLo(dir, lo);
    tmp_probdom.setHi(dir, hi);
    tmp_dom.setSmall(dir, 0);
    tmp_dom.setBig(dir, 0);
    m_slice_geom = amrex::Geometry(
        tmp_dom, tmp_probdom, Geom(lev).Coord(), Geom(lev).isPeriodic());
}

bool
Hipace::InSameTransverseCommunicator (int rank) const
{
    return rank/(m_numprocs_x*m_numprocs_y) == m_rank_z;
}

void
Hipace::InitData ()
{
    HIPACE_PROFILE("Hipace::InitData()");
    amrex::Print() << "HiPACE++ (" << Hipace::Version() << ")\n";

    amrex::Vector<amrex::IntVect> new_max_grid_size;
    for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
        amrex::IntVect mgs = maxGridSize(ilev);
        mgs[0] = mgs[1] = 1024000000; // disable domain decomposition in x and y directions
        new_max_grid_size.push_back(mgs);
    }
    SetMaxGridSize(new_max_grid_size);

    AmrCore::InitFromScratch(0.0); // function argument is time
    constexpr int lev = 0;
    m_multi_beam.InitData(geom[0]);
    m_multi_plasma.InitData(lev, m_slice_ba, m_slice_dm, m_slice_geom);
    m_adaptive_time_step.Calculate(m_dt, m_multi_beam, m_multi_plasma.maxDensity());
#ifdef AMREX_USE_MPI
    m_adaptive_time_step.WaitTimeStep(m_dt, m_comm_z);
    m_adaptive_time_step.NotifyTimeStep(m_dt, m_comm_z);
#endif
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
    DefineSliceGDB(ba, dm);
    m_fields.AllocData(lev, ba, dm, Geom(lev), m_slice_ba, m_slice_dm);
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
    HIPACE_PROFILE("Hipace::Evolve()");
    const int rank = amrex::ParallelDescriptor::MyProc();
    int const lev = 0;

    m_box_sorters.clear();
    m_multi_beam.sortParticlesByBox(m_box_sorters, boxArray(lev), geom[lev]);

    // now each rank starts with its own time step and writes to its own file. Highest rank starts with step 0
    for (int step = m_numprocs_z - 1 - m_rank_z; step < m_max_step; step += m_numprocs_z)
    {
#ifdef HIPACE_USE_OPENPMD
        if (m_output_period > 0) m_openpmd_writer.InitDiagnostics();
#endif

        if (m_verbose>=1) std::cout<<"Rank "<<rank<<" started  step "<<step<<" with dt = "<<m_dt<<'\n';

        ResetAllQuantities(lev);

        /* Store charge density of (immobile) ions into WhichSlice::RhoIons */
        m_multi_plasma.DepositNeutralizingBackground(m_fields, WhichSlice::RhoIons, geom[lev], lev);

        // Loop over longitudinal boxes on this rank, from head to tail
        for (int it = m_numprocs_z-1; it >= 0; --it)
        {
            Wait(step, it);

            m_box_sorters.clear();
            m_multi_beam.sortParticlesByBox(m_box_sorters, boxArray(lev), geom[lev]);
            m_leftmost_box_snd = std::min(leftmostBoxWithParticles(), m_leftmost_box_snd);

            const amrex::Box& bx = boxArray(lev)[it];
            m_fields.ResizeFDiagFAB(bx, lev);

            amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins;
            bins = m_multi_beam.findParticlesInEachSlice(lev, it, bx, geom[lev], m_box_sorters);

            for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
                SolveOneSlice(isl, lev, it, bins);
            };

            m_adaptive_time_step.Calculate(m_dt, m_multi_beam, m_multi_plasma.maxDensity(),
                                           it, m_box_sorters, false);

           // averaging predictor corrector loop diagnostics
           m_predcorr_avg_iterations /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));
           m_predcorr_avg_B_error /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));
#ifdef HIPACE_USE_OPENPMD
            WriteDiagnostics(step+1, it);
#else
            amrex::Print()<<"WARNING: In parallel runs, only openPMD supports dumping all time steps. \n";
#endif
            Notify(step, it);
        }

        // printing and resetting predictor corrector loop diagnostics
        if (m_verbose>=2) amrex::AllPrint()<<"Rank "<<rank<<": avg. number of iterations "
                                   << m_predcorr_avg_iterations << " avg. transverse B field error "
                                   << m_predcorr_avg_B_error << "\n";
        m_predcorr_avg_iterations = 0.;
        m_predcorr_avg_B_error = 0.;

        m_physical_time += m_dt;
    }
    // For consistency, decrement the physical time, so the last time step is like the others:
    // the time stored in the output file is the time for the fields. The beam is one time step
    // ahead.
    m_physical_time -= m_dt;

#ifdef HIPACE_USE_OPENPMD
    if (m_output_period > 0) m_openpmd_writer.reset();
#endif
}

void
Hipace::SolveOneSlice (int islice, int lev, const int ibox,
                       amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins)
{
    HIPACE_PROFILE("Hipace::SolveOneSlice()");
    // Between this push and the corresponding pop at the end of this
    // for loop, the parallelcontext is the transverse communicator
    amrex::ParallelContext::push(m_comm_xy);

    const amrex::Box& bx = boxArray(lev)[ibox];

    m_fields.getSlices(lev, WhichSlice::This).setVal(0.);

    m_multi_plasma.AdvanceParticles(m_fields, geom[lev], false, true, false, false, lev);

    amrex::MultiFab rho(m_fields.getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["rho"], 1);

    m_multi_plasma.DepositCurrent(
        m_fields, WhichSlice::This, false, true, true, true, geom[lev], lev);
    m_fields.AddRhoIons(lev);

    // need to exchange jx jy jz rho
    amrex::MultiFab j_slice(m_fields.getSlices(lev, WhichSlice::This),
                            amrex::make_alias, Comps[WhichSlice::This]["jx"], 4);
    j_slice.FillBoundary(Geom(lev).periodicity());

    m_fields.SolvePoissonExmByAndEypBx(Geom(lev), m_comm_xy, lev);

    m_grid_current.DepositCurrentSlice(m_fields, geom[lev], lev, islice);
    m_multi_beam.DepositCurrentSlice(m_fields, geom[lev], lev, islice, bx, bins, m_box_sorters,
                                     ibox, m_do_beam_jx_jy_deposition);

    j_slice.FillBoundary(Geom(lev).periodicity());

    m_fields.SolvePoissonEz(Geom(lev),lev);
    m_fields.SolvePoissonBz(Geom(lev), lev);

    /* Modifies Bx and By in the current slice
     * and the force terms of the plasma particles
     */
    PredictorCorrectorLoopToSolveBxBy(islice, lev);

    // Push beam particles
    m_multi_beam.AdvanceBeamParticlesSlice(m_fields, geom[lev], lev, islice, bx, bins, m_box_sorters, ibox);

    m_fields.FillDiagnostics(lev, islice);

    m_fields.ShiftSlices(lev);

    // After this, the parallel context is the full 3D communicator again
    amrex::ParallelContext::pop();
}

void
Hipace::ResetAllQuantities (int lev)
{
    HIPACE_PROFILE("Hipace::ResetAllQuantities()");
    m_multi_plasma.ResetParticles(lev, true);

    for (int islice=0; islice<WhichSlice::N; islice++) {
        m_fields.getSlices(lev, islice).setVal(0.);
    }
}

void
Hipace::PredictorCorrectorLoopToSolveBxBy (const int islice, const int lev)
{
    HIPACE_PROFILE("Hipace::PredictorCorrectorLoopToSolveBxBy()");

    amrex::Real relative_Bfield_error_prev_iter = 1.0;
    amrex::Real relative_Bfield_error = m_fields.ComputeRelBFieldError(
        m_fields.getSlices(lev, WhichSlice::Previous1),
        m_fields.getSlices(lev, WhichSlice::Previous1),
        m_fields.getSlices(lev, WhichSlice::Previous2),
        m_fields.getSlices(lev, WhichSlice::Previous2),
        Comps[WhichSlice::Previous1]["Bx"], Comps[WhichSlice::Previous1]["By"],
        Comps[WhichSlice::Previous2]["Bx"], Comps[WhichSlice::Previous2]["By"],
        Geom(lev));

    /* Guess Bx and By */
    m_fields.InitialBfieldGuess(relative_Bfield_error, m_predcorr_B_error_tolerance, lev);
    amrex::ParallelContext::push(m_comm_xy);
     // exchange ExmBy EypBx Ez Bx By Bz
    m_fields.getSlices(lev, WhichSlice::This).FillBoundary(Geom(lev).periodicity());
    amrex::ParallelContext::pop();

    /* creating temporary Bx and By arrays for the current and previous iteration */
    amrex::MultiFab Bx_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                            m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                            m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab By_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                            m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                            m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    Bx_iter.setVal(0.0);
    By_iter.setVal(0.0);
    amrex::MultiFab Bx_prev_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                                 m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                                 m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab::Copy(Bx_prev_iter, m_fields.getSlices(lev, WhichSlice::This),
                          Comps[WhichSlice::This]["Bx"], 0, 1, 0);
    amrex::MultiFab By_prev_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                                 m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                                 m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab::Copy(By_prev_iter, m_fields.getSlices(lev, WhichSlice::This),
                          Comps[WhichSlice::This]["By"], 0, 1, 0);

    /* creating aliases to the current in the next slice.
     * This needs to be reset after each push to the next slice */
    amrex::MultiFab jx_next(m_fields.getSlices(lev, WhichSlice::Next),
                            amrex::make_alias, Comps[WhichSlice::Next]["jx"], 1);
    amrex::MultiFab jy_next(m_fields.getSlices(lev, WhichSlice::Next),
                            amrex::make_alias, Comps[WhichSlice::Next]["jy"], 1);


    /* shift force terms, update force terms using guessed Bx and By */
    m_multi_plasma.AdvanceParticles( m_fields, geom[lev], false, false, true, true, lev);

    /* Begin of predictor corrector loop  */
    int i_iter = 0;
    /* resetting the initial B-field error for mixing between iterations */
    relative_Bfield_error = 1.0;
    while (( relative_Bfield_error > m_predcorr_B_error_tolerance )
           && ( i_iter < m_predcorr_max_iterations ))
    {
        i_iter++;
        m_predcorr_avg_iterations += 1.0;

        /* Push particles to the next slice */
        m_multi_plasma.AdvanceParticles(m_fields, geom[lev], true, true, false, false, lev);

        /* deposit current to next slice */
        m_multi_plasma.DepositCurrent(
            m_fields, WhichSlice::Next, true, true, false, false, geom[lev], lev);
        amrex::ParallelContext::push(m_comm_xy);
        // need to exchange jx jy jz rho
        amrex::MultiFab j_slice_next(m_fields.getSlices(lev, WhichSlice::Next),
                                     amrex::make_alias, Comps[WhichSlice::Next]["jx"], 4);
        j_slice_next.FillBoundary(Geom(lev).periodicity());
        amrex::ParallelContext::pop();

        /* Calculate Bx and By */
        m_fields.SolvePoissonBx(Bx_iter, Geom(lev), lev);
        m_fields.SolvePoissonBy(By_iter, Geom(lev), lev);

        relative_Bfield_error = m_fields.ComputeRelBFieldError(
            m_fields.getSlices(lev, WhichSlice::This),
            m_fields.getSlices(lev, WhichSlice::This),
            Bx_iter, By_iter,
            Comps[WhichSlice::This]["Bx"], Comps[WhichSlice::This]["By"],
            0, 0, Geom(lev));

        if (i_iter == 1) relative_Bfield_error_prev_iter = relative_Bfield_error;

        /* Mixing the calculated B fields to the actual B field and shifting iterated B fields */
        m_fields.MixAndShiftBfields(
            Bx_iter, Bx_prev_iter, Comps[WhichSlice::This]["Bx"], relative_Bfield_error,
            relative_Bfield_error_prev_iter, m_predcorr_B_mixing_factor, lev);
        m_fields.MixAndShiftBfields(
            By_iter, By_prev_iter, Comps[WhichSlice::This]["By"], relative_Bfield_error,
            relative_Bfield_error_prev_iter, m_predcorr_B_mixing_factor, lev);

        /* resetting current in the next slice to clean temporarily used current*/
        jx_next.setVal(0.);
        jy_next.setVal(0.);

        amrex::ParallelContext::push(m_comm_xy);
         // exchange Bx By
        m_fields.getSlices(lev, WhichSlice::This).FillBoundary(Geom(lev).periodicity());
        amrex::ParallelContext::pop();

        /* Update force terms using the calculated Bx and By */
        m_multi_plasma.AdvanceParticles(m_fields, geom[lev], false, false, true, false, lev);

        /* Shift relative_Bfield_error values */
        relative_Bfield_error_prev_iter = relative_Bfield_error;
    } /* end of predictor corrector loop */

    /* resetting the particle position after they have been pushed to the next slice */
    m_multi_plasma.ResetParticles(lev);

    if (relative_Bfield_error > 10. && m_predcorr_B_error_tolerance > 0.)
    {
        amrex::Print() << "WARNING: Predictor corrector loop may have diverged!\n"
                     "Re-try by adjusting the following paramters in the input script:\n"
                     "- lower mixing factor: hipace.predcorr_B_mixing_factor "
                     "(hidden default: 0.1) \n"
                     "- lower B field error tolerance: hipace.predcorr_B_error_tolerance"
                     " (hidden default: 0.04)\n"
                     "- higher number of iterations in the pred. cor. loop:"
                     "hipace.predcorr_max_iterations (hidden default: 5)\n"
                     "- higher longitudinal resolution";
    }

    // adding relative B field error for diagnostic
    m_predcorr_avg_B_error += relative_Bfield_error;
    if (m_verbose >= 2) amrex::Print()<<"islice: " << islice << " n_iter: "<<i_iter<<
                            " relative B field error: "<<relative_Bfield_error<< "\n";
}

void
Hipace::Wait (const int step, int it)
{
    HIPACE_PROFILE("Hipace::Wait()");
#ifdef AMREX_USE_MPI
    if (step == 0) return;

    const int nbeams = m_multi_beam.get_nbeams();
    // 1 element per beam species, and 1 for
    // the index of leftmost box with beam particles.
    const int nint = nbeams + 1;
    amrex::Vector<int> np_rcv(nint, 0);
    if (it < m_leftmost_box_rcv && it < m_numprocs_z - 1 && m_skip_empty_comms){
        if (m_verbose >= 2){
            amrex::AllPrint()<<"rank "<<m_rank_z<<" step "<<step<<" box "<<it<<": SKIP RECV!\n";
        }
        return;
    }

    // Receive particle counts
    {
        MPI_Status status;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0
        MPI_Recv(np_rcv.dataPtr(), nint,
                 amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
                 (m_rank_z+1)%m_numprocs_z, ncomm_z_tag, m_comm_z, &status);
    }
    m_leftmost_box_rcv = std::min(np_rcv[nbeams], m_leftmost_box_rcv);

    // Receive beam particles.
    {
        const amrex::Long np_total = std::accumulate(np_rcv.begin(), np_rcv.begin()+nbeams, 0);
        if (np_total == 0) return;
        const amrex::Long psize = sizeof(BeamParticleContainer::SuperParticleType);
        const amrex::Long buffer_size = psize*np_total;
        auto recv_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

        MPI_Status status;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0
        MPI_Recv(recv_buffer, buffer_size,
                 amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                 (m_rank_z+1)%m_numprocs_z, pcomm_z_tag, m_comm_z, &status);

        int offset_beam = 0;
        for (int ibeam = 0; ibeam < nbeams; ibeam++){
            auto& ptile = m_multi_beam.getBeam(ibeam);
            const int np = np_rcv[ibeam];

            auto old_size = ptile.numParticles();
            auto new_size = old_size + np;
            ptile.resize(new_size);
            const auto ptd = ptile.getParticleTileData();

            const amrex::Gpu::DeviceVector<int> comm_real(m_multi_beam.NumRealComps(), 1);
            const amrex::Gpu::DeviceVector<int> comm_int (m_multi_beam.NumIntComps(),  1);
            const auto p_comm_real = comm_real.data();
            const auto p_comm_int = comm_int.data();

#ifdef AMREX_USE_GPU
            if (amrex::Gpu::inLaunchRegion() && np > 0) {
                int const np_per_block = 128;
                int const nblocks = (np+np_per_block-1)/np_per_block;
                std::size_t const shared_mem_bytes = np_per_block * psize;
                // NOTE - TODO DPC++
                amrex::launch(
                    nblocks, np_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
                    [=] AMREX_GPU_DEVICE () noexcept
                    {
                        amrex::Gpu::SharedMemory<char> gsm;
                        char* const shared = gsm.dataPtr();

                        // Copy packed data from recv_buffer (in pinned memory) to shared memory
                        const int i = blockDim.x*blockIdx.x+threadIdx.x;
                        const unsigned int m = threadIdx.x;
                        const unsigned int mend = amrex::min<unsigned int>
                            (blockDim.x, np-blockDim.x*blockIdx.x);
                        for (unsigned int index = m;
                             index < mend*psize/sizeof(double); index += blockDim.x) {
                            const double *csrc = (double *)
                                (recv_buffer+offset_beam*psize+blockDim.x*blockIdx.x*psize);
                            double *cdest = (double *)shared;
                            cdest[index] = csrc[index];
                        }

                        __syncthreads();
                        // Unpack in shared memory, and move to device memory
                        if (i < np) {
                            ptd.unpackParticleData(
                                shared, m*psize, i+old_size, p_comm_real, p_comm_int);
                        }
                    });
            } else
#endif
            {
                for (int i = 0; i < np; ++i)
                {
                    ptd.unpackParticleData(
                        recv_buffer+offset_beam*psize, i*psize, i+old_size, p_comm_real, p_comm_int);
                }
            }
            offset_beam += np;
        }

        amrex::Gpu::Device::synchronize();
        amrex::The_Pinned_Arena()->free(recv_buffer);
    }

#endif
}

void
Hipace::Notify (const int step, const int it)
{
    HIPACE_PROFILE("Hipace::Notify()");
    // Send from slices 2 and 3 (or main MultiFab's first two valid slabs) to receiver's slices 2
    // and 3.

#ifdef AMREX_USE_MPI
    NotifyFinish(); // finish the previous send

    const int nbeams = m_multi_beam.get_nbeams();
    const int nint = nbeams + 1;

    // last step does not need to send anything, but needs to resize to remove slipped particles
    if (step == m_max_step -1 )
    {
        for (int ibeam = 0; ibeam < nbeams; ibeam++){
            const int offset_box = m_box_sorters[ibeam].boxOffsetsPtr()[it];
            auto& ptile = m_multi_beam.getBeam(ibeam);
            ptile.resize(offset_box);
        }
        return;
    }

    m_leftmost_box_snd = std::min(m_leftmost_box_snd, m_leftmost_box_rcv);
    if (it < m_leftmost_box_snd && it < m_numprocs_z - 1 && m_skip_empty_comms){
        if (m_verbose >= 2){
            amrex::AllPrint()<<"rank "<<m_rank_z<<" step "<<step<<" box "<<it<<": SKIP SEND!\n";
        }
        return;
    }

    // 1 element per beam species, and 1 for the index of leftmost box with beam particles.
    m_np_snd.resize(nint);

    for (int ibeam = 0; ibeam < nbeams; ++ibeam)
    {
        m_np_snd[ibeam] = m_box_sorters[ibeam].boxCountsPtr()[it];
    }
    m_np_snd[nbeams] = m_leftmost_box_snd;

    // Each rank sends data downstream, except rank 0 who sends data to m_numprocs_z-1
    MPI_Isend(m_np_snd.dataPtr(), nint, amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
              (m_rank_z-1+m_numprocs_z)%m_numprocs_z, ncomm_z_tag, m_comm_z, &m_nsend_request);

    // Send beam particles. Currently only one tile.
    {
        const amrex::Long np_total = std::accumulate(m_np_snd.begin(), m_np_snd.begin()+nbeams, 0);
        if (np_total == 0) return;
        const amrex::Long psize = sizeof(BeamParticleContainer::SuperParticleType);
        const amrex::Long buffer_size = psize*np_total;
        m_psend_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

        int offset_beam = 0;
        for (int ibeam = 0; ibeam < nbeams; ibeam++){
            const int offset_box = m_box_sorters[ibeam].boxOffsetsPtr()[it];
            const amrex::Long np = m_np_snd[ibeam];

            auto& ptile = m_multi_beam.getBeam(ibeam);
            const auto ptd = ptile.getConstParticleTileData();

            const amrex::Gpu::DeviceVector<int> comm_real(m_multi_beam.NumRealComps(), 1);
            const amrex::Gpu::DeviceVector<int> comm_int (m_multi_beam.NumIntComps(),  1);
            const auto p_comm_real = comm_real.data();
            const auto p_comm_int = comm_int.data();
            const auto p_psend_buffer = m_psend_buffer + offset_beam*psize;
#ifdef AMREX_USE_GPU
            if (amrex::Gpu::inLaunchRegion() && np > 0) {
                const int np_per_block = 128;
                const int nblocks = (np+np_per_block-1)/np_per_block;
                const std::size_t shared_mem_bytes = np_per_block * psize;
                // NOTE - TODO DPC++
                amrex::launch(
                    nblocks, np_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
                    [=] AMREX_GPU_DEVICE () noexcept
                    {
                        amrex::Gpu::SharedMemory<char> gsm;
                        char* const shared = gsm.dataPtr();

                        // Pack particles from device memory to shared memory
                        const int i = blockDim.x*blockIdx.x+threadIdx.x;
                        const unsigned int m = threadIdx.x;
                        const unsigned int mend = amrex::min<unsigned int>(blockDim.x, np-blockDim.x*blockIdx.x);
                        if (i < np) {
                            ptd.packParticleData(shared, offset_box+i, m*psize, p_comm_real, p_comm_int);
                        }

                        __syncthreads();

                        // Copy packed particles from shared memory to psend_buffer in pinned memory
                        for (unsigned int index = m;
                             index < mend*psize/sizeof(double); index += blockDim.x) {
                            const double *csrc = (double *)shared;
                            double *cdest = (double *)(p_psend_buffer+blockDim.x*blockIdx.x*psize);
                            cdest[index] = csrc[index];
                        }
                    });
            } else
#endif
            {
                for (int i = 0; i < np; ++i)
                {
                    ptd.packParticleData(p_psend_buffer, offset_box+i, i*psize, p_comm_real, p_comm_int);
                }
            }
            amrex::Gpu::Device::synchronize();

            ptile.resize(offset_box);
            offset_beam += np;
        } // here
        // Each rank sends data downstream, except rank 0 who sends data to m_numprocs_z-1
        MPI_Isend(m_psend_buffer, buffer_size, amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                  (m_rank_z-1+m_numprocs_z)%m_numprocs_z, pcomm_z_tag, m_comm_z, &m_psend_request);
    }
#endif
}

void
Hipace::NotifyFinish ()
{
#ifdef AMREX_USE_MPI
    if (m_np_snd.size() > 0) {
        MPI_Status status;
        MPI_Wait(&m_nsend_request, &status);
        m_np_snd.resize(0);
    }
    if (m_psend_buffer) {
        MPI_Status status;
        MPI_Wait(&m_psend_request, &status);
        amrex::The_Pinned_Arena()->free(m_psend_buffer);
        m_psend_buffer = nullptr;
    }
#endif
}

void
Hipace::WriteDiagnostics (int output_step, const int it)
{
    HIPACE_PROFILE("Hipace::WriteDiagnostics()");

    // Dump every m_output_period steps and after last step
    if (m_output_period < 0 ||
        (!(output_step == m_max_step) && output_step % m_output_period != 0) ) return;

    // Write fields
    const std::string filename = amrex::Concatenate("plt", output_step);
    // assumption: same order as in struct enum Field Comps
    const amrex::Vector< std::string > varnames = m_fields.getDiagComps();

    amrex::Vector<std::string> rfs;

#ifdef HIPACE_USE_OPENPMD
    constexpr int lev = 0;
    m_openpmd_writer.WriteDiagnostics(m_fields.getDiagF(), m_multi_beam, m_fields.getDiagGeom(),
                        m_physical_time, output_step, lev, m_fields.getDiagSliceDir(), varnames,
                        it, m_box_sorters);
#else
    constexpr int nlev = 1;
    const amrex::IntVect local_ref_ratio {1, 1, 1};

    amrex::WriteMultiLevelPlotfile(
        filename, nlev, amrex::GetVecOfConstPtrs(m_fields.getDiagF()), varnames,
        m_fields.getDiagGeom(), m_physical_time, {output_step}, {local_ref_ratio},
        "HyperCLaw-V1.1", "Level_", "Cell", rfs);

    // Write beam particles
    m_multi_beam.WritePlotFile(filename);
#endif
}

std::string
Hipace::Version ()
{
#ifdef HIPACE_GIT_VERSION
    return std::string(HIPACE_GIT_VERSION);
#else
    return std::string("Unknown");
#endif
}

int
Hipace::leftmostBoxWithParticles () const
{
    int boxid = m_numprocs_z;
    for(const auto& box_sorter : m_box_sorters){
        boxid = std::min(box_sorter.leftmostBoxWithParticles(), boxid);
    }
    return boxid;
}
