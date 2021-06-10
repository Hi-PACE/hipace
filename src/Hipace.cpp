#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "particles/BinSort.H"
#include "particles/BoxSort.H"
#include "utils/IOUtil.H"
#include "particles/pusher/GetAndSetPosition.H"

#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>
#ifdef AMREX_USE_LINEAR_SOLVERS
#  include <AMReX_MLALaplacian.H>
#  include <AMReX_MLMG.H>
#endif

#include <algorithm>
#include <memory>

#ifdef AMREX_USE_MPI
namespace {
    constexpr int ncomm_z_tag = 1001;
    constexpr int pcomm_z_tag = 1002;
    constexpr int ncomm_z_tag_ghost = 1003;
    constexpr int pcomm_z_tag_ghost = 1004;
    constexpr int tcomm_z_tag = 1005;
}
#endif

Hipace* Hipace::m_instance = nullptr;

int Hipace::m_max_step = 0;
amrex::Real Hipace::m_dt = 0.0;
bool Hipace::m_normalized_units = false;
amrex::Real Hipace::m_physical_time = 0.0;
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
amrex::Real Hipace::m_MG_tolerance_rel = 1.e-4;
amrex::Real Hipace::m_MG_tolerance_abs = 0.;

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
    m_multi_plasma(this),
    m_diags(this->maxLevel()+1)
{
    m_instance = this;

    amrex::ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
    pp.query("max_step", m_max_step);

#ifndef AMREX_USE_GPU
    int seed;
    if (pp.query("random_seed", seed)) amrex::ResetRandomSeed(seed);
#endif

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
    m_numprocs_z = amrex::ParallelDescriptor::NProcs() / (m_numprocs_x*m_numprocs_y);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_z <= m_max_step+1,
                                     "Please use more or equal time steps than number of ranks");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_x*m_numprocs_y*m_numprocs_z
                                     == amrex::ParallelDescriptor::NProcs(),
                                     "Check hipace.numprocs_x and hipace.numprocs_y");
    pph.query("boxes_in_z", m_boxes_in_z);
    if (m_boxes_in_z > 1) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_numprocs_z == 1,
                            "Multiple boxes per rank only implemented for one rank.");
    pph.query("depos_order_xy", m_depos_order_xy);
    pph.query("depos_order_z", m_depos_order_z);
    pph.query("predcorr_B_error_tolerance", m_predcorr_B_error_tolerance);
    pph.query("predcorr_max_iterations", m_predcorr_max_iterations);
    pph.query("predcorr_B_mixing_factor", m_predcorr_B_mixing_factor);
    pph.query("output_period", m_output_period);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_output_period != 0,
                                     "To avoid output, please use output_period = -1.");
    pph.query("beam_injection_cr", m_beam_injection_cr);
    pph.query("do_beam_jx_jy_deposition", m_do_beam_jx_jy_deposition);
    pph.query("do_device_synchronize", m_do_device_synchronize);
    pph.query("external_ExmBy_slope", m_external_ExmBy_slope);
    pph.query("external_Ez_slope", m_external_Ez_slope);
    pph.query("external_Ez_uniform", m_external_Ez_uniform);
    std::string solver = "predictor-corrector";
    pph.query("bxby_solver", solver);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solver == "predictor-corrector" ||
        solver == "explicit",
        "hipace.bxby_solver must be predictor-corrector or explicit");
    if (solver == "explicit") m_explicit = true;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_explicit && !m_multi_plasma.AllSpeciesNeutralizeBackground()),
        "Ion motion with explicit solver is not implemented, need to use neutralize_background");

    pph.query("MG_tolerance_rel", m_MG_tolerance_rel);
    pph.query("MG_tolerance_abs", m_MG_tolerance_abs);

    if (maxLevel() > 0) {
        AMREX_ALWAYS_ASSERT(maxLevel() < 2);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        pph.get("patch_lo", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) patch_lo[idim] = loc_array[idim];
        pph.get("patch_hi", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) patch_hi[idim] = loc_array[idim];
    }

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
    NotifyFinish(true);
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
    // Set the lo and hi of domain and probdomain in the z direction
    amrex::RealBox tmp_probdom = Geom(lev).ProbDomain();
    amrex::Box tmp_dom = Geom(lev).Domain();
    const amrex::Real dz = Geom(lev).CellSize(Direction::z);
    const amrex::Real hi = Geom(lev).ProbHi(Direction::z);
    const amrex::Real lo = hi - dz;
    tmp_probdom.setLo(Direction::z, lo);
    tmp_probdom.setHi(Direction::z, hi);
    tmp_dom.setSmall(Direction::z, 0);
    tmp_dom.setBig(Direction::z, 0);
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
    m_multi_beam.InitData(geom[lev]);
    m_multi_plasma.InitData(lev, m_slice_ba, m_slice_dm, m_slice_geom, geom[lev]);
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
        const amrex::IntVect ncells_global = Geom(lev).Domain().length();
        const amrex::IntVect box_size = ba[0].length();  // Uniform box size
        const int nboxes_x = m_numprocs_x;
        const int nboxes_y = m_numprocs_y;
        const int nboxes_z = (m_boxes_in_z == 1) ? ncells_global[2] / box_size[2] : m_boxes_in_z;
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
    // Note: we pass ba[0] as a dummy box, it will be resized properly in the loop over boxes in Evolve
    m_diags.AllocData(lev, ba[0], Comps[WhichSlice::This]["N"], Geom(lev));
    m_fields.AllocData(lev, ba, dm, Geom(lev), m_slice_ba, m_slice_dm);
}

void
Hipace::ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/)
{
    using namespace amrex::literals;
    const amrex::Real* problo = Geom(lev).ProbLo();
    const amrex::Real* dx = Geom(lev).CellSize();

    for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi)
    {
        auto& fab = tags[mfi];
        const amrex::Box& bx = fab.box();
        for (amrex::BoxIterator bi(bx); bi.ok(); ++bi)
        {
            const amrex::IntVect& cell = bi();
            amrex::RealVect pos {AMREX_D_DECL((cell[0]+0.5_rt)*dx[0]+problo[0],
                                       (cell[1]+0.5_rt)*dx[1]+problo[1],
                                       (cell[2]+0.5_rt)*dx[2]+problo[2])};
            if (pos > patch_lo && pos < patch_hi) {
                fab(cell) = amrex::TagBox::SET;
            }
        }
    }
}

void
Hipace::PostProcessBaseGrids (amrex::BoxArray& ba0) const
{
    // This is called by AmrCore::InitFromScratch.
    // The BoxArray made by AmrCore is not what we want.  We will replace it with our own.
    const int lev = 0;
    const amrex::IntVect ncells_global = Geom(lev).Domain().length();
    amrex::IntVect box_size{ncells_global[0] / m_numprocs_x,
                            ncells_global[1] / m_numprocs_y,
                            ncells_global[2] / m_boxes_in_z};
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_size[0]*m_numprocs_x == ncells_global[0],
                                     "# of cells in x-direction is not divisible by hipace.numprocs_x");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_size[1]*m_numprocs_y == ncells_global[1],
                                     "# of cells in y-direction is not divisible by hipace.numprocs_y");

    if (m_boxes_in_z == 1) {
        box_size[2] = ncells_global[2] / m_numprocs_z;
    }

    const int nboxes_x = m_numprocs_x;
    const int nboxes_y = m_numprocs_y;
    const int nboxes_z = (m_boxes_in_z == 1) ? ncells_global[2] / box_size[2] : m_boxes_in_z;
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
    for (int step = m_numprocs_z - 1 - m_rank_z; step <= m_max_step; step += m_numprocs_z)
    {
#ifdef HIPACE_USE_OPENPMD
        m_openpmd_writer.InitDiagnostics(step, m_output_period, m_max_step);
#endif

        if (m_verbose>=1) std::cout<<"Rank "<<rank<<" started  step "<<step<<" with dt = "<<m_dt<<'\n';

        ResetAllQuantities(lev);

        /* Store charge density of (immobile) ions into WhichSlice::RhoIons */
        m_multi_plasma.DepositNeutralizingBackground(m_fields, WhichSlice::RhoIons, geom[lev], lev);

        // Loop over longitudinal boxes on this rank, from head to tail
        const int n_boxes = (m_boxes_in_z == 1) ? m_numprocs_z : m_boxes_in_z;
        for (int it = n_boxes-1; it >= 0; --it)
        {
            Wait(step, it);

            m_box_sorters.clear();

            m_multi_beam.sortParticlesByBox(m_box_sorters, boxArray(lev), geom[lev]);
            m_leftmost_box_snd = std::min(leftmostBoxWithParticles(), m_leftmost_box_snd);

            WriteDiagnostics(step, it, OpenPMDWriterCallType::beams);

            m_multi_beam.StoreNRealParticles();
            // Copy particles in box it-1 in the ghost buffer.
            // This handles both beam initialization and particle slippage.
            if (it>0) m_multi_beam.PackLocalGhostParticles(it-1, m_box_sorters);

            const amrex::Box& bx = boxArray(lev)[it];
            ResizeFDiagFAB(bx, lev);

            amrex::Vector<BeamBins> bins;
            bins = m_multi_beam.findParticlesInEachSlice(lev, it, bx, geom[lev], m_box_sorters);
            AMREX_ALWAYS_ASSERT( bx.bigEnd(Direction::z) >= bx.smallEnd(Direction::z) + 2 );
            // Solve head slice
            SolveOneSlice(bx.bigEnd(Direction::z), lev, it, bins);
            // Notify ghost slice
            if (it<m_numprocs_z-1) Notify(step, it, bins, true);
            // Solve central slices
            for (int isl = bx.bigEnd(Direction::z)-1; isl > bx.smallEnd(Direction::z); --isl){
                SolveOneSlice(isl, lev, it, bins);
            };
            // Receive ghost slice
            if (it>0) Wait(step, it, true);
            CheckGhostSlice(it);
            // Solve tail slice. Consume ghost particles.
            SolveOneSlice(bx.smallEnd(Direction::z), lev, it, bins);
            // Delete ghost particles
            m_multi_beam.RemoveGhosts();

            m_adaptive_time_step.Calculate(m_dt, m_multi_beam, m_multi_plasma.maxDensity(),
                                           it, m_box_sorters, false);

            // averaging predictor corrector loop diagnostics
            m_predcorr_avg_iterations /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));
            m_predcorr_avg_B_error /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));

            WriteDiagnostics(step, it, OpenPMDWriterCallType::fields);

            Notify(step, it, bins);
        }

        // printing and resetting predictor corrector loop diagnostics
        if (m_verbose>=2) amrex::AllPrint()<<"Rank "<<rank<<": avg. number of iterations "
                                   << m_predcorr_avg_iterations << " avg. transverse B field error "
                                   << m_predcorr_avg_B_error << "\n";
        m_predcorr_avg_iterations = 0.;
        m_predcorr_avg_B_error = 0.;

        m_physical_time += m_dt;
    }

#ifdef HIPACE_USE_OPENPMD
    if (m_output_period > 0) m_openpmd_writer.reset();
#endif
}

void
Hipace::SolveOneSlice (int islice, int lev, const int ibox,
                       amrex::Vector<BeamBins>& bins)
{
    HIPACE_PROFILE("Hipace::SolveOneSlice()");
    // Between this push and the corresponding pop at the end of this
    // for loop, the parallelcontext is the transverse communicator
    amrex::ParallelContext::push(m_comm_xy);

    const amrex::Box& bx = boxArray(lev)[ibox];

    if (m_explicit) {
        // Set all quantities to 0 except Bx and By: the previous slice serves as initial guess.
        const int ibx = Comps[WhichSlice::This]["Bx"];
        const int iby = Comps[WhichSlice::This]["By"];
        const int nc = Comps[WhichSlice::This]["N"];
        AMREX_ALWAYS_ASSERT( iby == ibx+1 );
        m_fields.getSlices(lev, WhichSlice::This).setVal(0., 0, ibx);
        m_fields.getSlices(lev, WhichSlice::This).setVal(0., iby+1, nc-iby-1);
    } else {
        m_fields.getSlices(lev, WhichSlice::This).setVal(0.);
    }

    if (!m_explicit) m_multi_plasma.AdvanceParticles(m_fields, geom[lev], false,
                                                     true, false, false, lev);

    amrex::MultiFab rho(m_fields.getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["rho"], 1);

    m_multi_plasma.DepositCurrent(
        m_fields, WhichSlice::This, false, true, true, true, m_explicit, geom[lev], lev);

        if (m_explicit){
            amrex::MultiFab j_slice_next(m_fields.getSlices(lev, WhichSlice::Next),
                                         amrex::make_alias, Comps[WhichSlice::Next]["jx"], 4);
            j_slice_next.setVal(0.);
            m_multi_beam.DepositCurrentSlice(m_fields, geom[lev], lev, islice, bx, bins, m_box_sorters,
                                             ibox, m_do_beam_jx_jy_deposition, WhichSlice::Next);
            m_fields.AddBeamCurrents(lev, WhichSlice::Next);
            // need to exchange jx jy jx_beam jy_beam
            j_slice_next.FillBoundary(Geom(lev).periodicity());
        }

    m_fields.AddRhoIons(lev);

    // need to exchange jx jy jz jx_beam jy_beam jz_beam rho
    // Assert that the order of the transverse currents and charge density is correct. This order is
    // also required in the FillBoundary call on the next slice in the predictor-corrector loop, as
    // well as in the shift slices.
    const int ijx = Comps[WhichSlice::This]["jx"];
    const int ijx_beam = Comps[WhichSlice::This]["jx_beam"];
    const int ijy = Comps[WhichSlice::This]["jy"];
    const int ijy_beam = Comps[WhichSlice::This]["jy_beam"];
    const int ijz = Comps[WhichSlice::This]["jz"];
    const int ijz_beam = Comps[WhichSlice::This]["jz_beam"];
    const int irho = Comps[WhichSlice::This]["rho"];
    AMREX_ALWAYS_ASSERT( ijx_beam == ijx+1 && ijy == ijx+2 && ijy_beam == ijx+3 && ijz == ijx+4 &&
                         ijz_beam == ijx+5 && irho == ijx+6 );
    amrex::MultiFab j_slice(m_fields.getSlices(lev, WhichSlice::This),
                            amrex::make_alias, Comps[WhichSlice::This]["jx"], 7);
    j_slice.FillBoundary(Geom(lev).periodicity());

    m_fields.SolvePoissonExmByAndEypBx(Geom(lev), m_comm_xy, lev);

    m_grid_current.DepositCurrentSlice(m_fields, geom[lev], lev, islice);
    m_multi_beam.DepositCurrentSlice(m_fields, geom[lev], lev, islice, bx, bins, m_box_sorters,
                                     ibox, m_do_beam_jx_jy_deposition, WhichSlice::This);
    m_fields.AddBeamCurrents(lev, WhichSlice::This);

    j_slice.FillBoundary(Geom(lev).periodicity());

    m_fields.SolvePoissonEz(Geom(lev),lev);
    m_fields.SolvePoissonBz(Geom(lev), lev);

    // Modifies Bx and By in the current slice and the force terms of the plasma particles
    if (m_explicit){
        m_fields.AddRhoIons(lev, true);
        ExplicitSolveBxBy(lev);
        m_multi_plasma.AdvanceParticles( m_fields, geom[lev], false, true, true, true, lev);
        m_fields.AddRhoIons(lev);
    } else {
        PredictorCorrectorLoopToSolveBxBy(islice, lev, bx, bins, ibox);
    }

    // Push beam particles
    m_multi_beam.AdvanceBeamParticlesSlice(m_fields, geom[lev], lev, islice, bx, bins, m_box_sorters, ibox);

    FillDiagnostics(lev, islice);

    m_fields.ShiftSlices(lev);

    m_multi_plasma.DoFieldIonization(lev, geom[lev], m_fields);

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
Hipace::ExplicitSolveBxBy (const int lev)
{
    HIPACE_PROFILE("Hipace::ExplicitSolveBxBy()");
    amrex::ParallelContext::push(m_comm_xy);
    using namespace amrex::literals;

    const int isl = WhichSlice::This;
    amrex::MultiFab& slicemf = m_fields.getSlices(lev, isl);
    const int nsl = WhichSlice::Next;
    amrex::MultiFab& nslicemf = m_fields.getSlices(lev, nsl);
    const int psl = WhichSlice::Previous1;
    amrex::MultiFab& pslicemf = m_fields.getSlices(lev, psl);
    const amrex::BoxArray ba = slicemf.boxArray();
    const amrex::DistributionMapping dm = slicemf.DistributionMap();
    const amrex::IntVect ngv = slicemf.nGrowVect();

    // Later this should have only 1 component, but we have 2 for now, with always the same values.
    amrex::MultiFab Mult(ba, dm, 2, ngv);
    amrex::MultiFab S(ba, dm, 2, ngv);
    Mult.setVal(0.);
    S.setVal(0.);

    const amrex::MultiFab Rho(slicemf, amrex::make_alias, Comps[isl]["rho"    ], 1);
    const amrex::MultiFab Jx (slicemf, amrex::make_alias, Comps[isl]["jx"     ], 1);
    const amrex::MultiFab Jy (slicemf, amrex::make_alias, Comps[isl]["jy"     ], 1);
    const amrex::MultiFab Jxb(slicemf, amrex::make_alias, Comps[isl]["jx_beam"], 1);
    const amrex::MultiFab Jyb(slicemf, amrex::make_alias, Comps[isl]["jy_beam"], 1);
    const amrex::MultiFab Jxx(slicemf, amrex::make_alias, Comps[isl]["jxx"    ], 1);
    const amrex::MultiFab Jxy(slicemf, amrex::make_alias, Comps[isl]["jxy"    ], 1);
    const amrex::MultiFab Jyy(slicemf, amrex::make_alias, Comps[isl]["jyy"    ], 1);
    const amrex::MultiFab Jz (slicemf, amrex::make_alias, Comps[isl]["jz"     ], 1);
    const amrex::MultiFab Jzb(slicemf, amrex::make_alias, Comps[isl]["jz_beam"], 1);
    const amrex::MultiFab Psi(slicemf, amrex::make_alias, Comps[isl]["Psi"    ], 1);
    const amrex::MultiFab Bz (slicemf, amrex::make_alias, Comps[isl]["Bz"     ], 1);
    const amrex::MultiFab Ez (slicemf, amrex::make_alias, Comps[isl]["Ez"     ], 1);
    const amrex::MultiFab prev_Jxb(pslicemf, amrex::make_alias, Comps[psl]["jx_beam"], 1);
    const amrex::MultiFab next_Jxb(nslicemf, amrex::make_alias, Comps[nsl]["jx_beam"], 1);
    const amrex::MultiFab prev_Jyb(pslicemf, amrex::make_alias, Comps[psl]["jy_beam"], 1);
    const amrex::MultiFab next_Jyb(nslicemf, amrex::make_alias, Comps[nsl]["jy_beam"], 1);
    amrex::MultiFab BxBy (slicemf, amrex::make_alias, Comps[isl]["Bx" ], 2);

    // preparing conversion to normalized units, if applicable
    PhysConst pc = m_phys_const;
    const amrex::Real n0 = m_multi_plasma.maxDensity();
    const amrex::Real omegap = std::sqrt(n0 * pc.q_e*pc.q_e/(pc.m_e*pc.ep0));
    const amrex::Real kp = omegap/pc.c;
    const amrex::Real kpinv = 1./kp;
    const amrex::Real E0 = omegap * pc.m_e * pc.c / pc.q_e;

    // dx, dy, dz in normalized units
    const amrex::Real dx = Geom(lev).CellSize(Direction::x)/kpinv;
    const amrex::Real dy = Geom(lev).CellSize(Direction::y)/kpinv;
    const amrex::Real dz = Geom(lev).CellSize(Direction::z)/kpinv;

    // transforming BxBy array to normalized units for use as initial guess
    BxBy.mult(pc.c/E0);

    for ( amrex::MFIter mfi(Bz, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){

        // add enough guard cells to enable transverse derivatives
        amrex::Box const& bx = mfi.growntilebox({1,1,0});

        amrex::Array4<amrex::Real const> const & rho = Rho.array(mfi);
        amrex::Array4<amrex::Real const> const & jx  = Jx .array(mfi);
        amrex::Array4<amrex::Real const> const & jy  = Jy .array(mfi);
        amrex::Array4<amrex::Real const> const & jxb = Jxb.array(mfi);
        amrex::Array4<amrex::Real const> const & jyb = Jyb.array(mfi);
        amrex::Array4<amrex::Real const> const & jxx = Jxx.array(mfi);
        amrex::Array4<amrex::Real const> const & jxy = Jxy.array(mfi);
        amrex::Array4<amrex::Real const> const & jyy = Jyy.array(mfi);
        amrex::Array4<amrex::Real const> const & jz  = Jz .array(mfi);
        amrex::Array4<amrex::Real const> const & jzb = Jzb.array(mfi);
        amrex::Array4<amrex::Real const> const & psi = Psi.array(mfi);
        amrex::Array4<amrex::Real const> const & bz  = Bz.array(mfi);
        amrex::Array4<amrex::Real const> const & ez  = Ez.array(mfi);
        amrex::Array4<amrex::Real const> const & next_jxb = next_Jxb.array(mfi);
        amrex::Array4<amrex::Real const> const & prev_jxb = prev_Jxb.array(mfi);
        amrex::Array4<amrex::Real const> const & next_jyb = next_Jyb.array(mfi);
        amrex::Array4<amrex::Real const> const & prev_jyb = prev_Jyb.array(mfi);
        amrex::Array4<amrex::Real> const & mult = Mult.array(mfi);
        amrex::Array4<amrex::Real> const & s = S.array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const amrex::Real dx_jxy = (jxy(i+1,j,k)-jxy(i-1,j,k))/(2._rt*dx);
                const amrex::Real dx_jxx = (jxx(i+1,j,k)-jxx(i-1,j,k))/(2._rt*dx);
                const amrex::Real dx_jz  = (jz (i+1,j,k)-jz (i-1,j,k))/(2._rt*dx);
                const amrex::Real dx_psi = (psi(i+1,j,k)-psi(i-1,j,k))/(2._rt*dx);

                const amrex::Real dy_jyy = (jyy(i,j+1,k)-jyy(i,j-1,k))/(2._rt*dy);
                const amrex::Real dy_jxy = (jxy(i,j+1,k)-jxy(i,j-1,k))/(2._rt*dy);
                const amrex::Real dy_jz  = (jz (i,j+1,k)-jz (i,j-1,k))/(2._rt*dy);
                const amrex::Real dy_psi = (psi(i,j+1,k)-psi(i,j-1,k))/(2._rt*dy);

                const amrex::Real dz_jxb = (prev_jxb(i,j,k)-next_jxb(i,j,k))/(2._rt*dz);
                const amrex::Real dz_jyb = (prev_jyb(i,j,k)-next_jyb(i,j,k))/(2._rt*dz);

                // Store (i,j,k) cell value in local variable.
                // All quantities are converted to normalized units, if applicable
                // NOTE: a few -1 factors are added here, due to discrepancy in definitions between
                // WAND-PIC and HiPACE++:
                //   n* and j are defined from ne in WAND-PIC and from rho in hipace++.
                const amrex::Real cne     = - rho(i,j,k) / n0 / pc.q_e ;
                const amrex::Real cjzp    = - (jz(i,j,k) - jzb(i,j,k)) / n0 / pc.q_e / pc.c;
                const amrex::Real cjxp    = - (jx(i,j,k) - jxb(i,j,k)) / n0 / pc.q_e / pc.c;
                const amrex::Real cjyp    = - (jy(i,j,k) - jyb(i,j,k)) / n0 / pc.q_e / pc.c;
                const amrex::Real cpsi    =   psi(i,j,k) * pc.q_e / (pc.m_e * pc.c * pc.c);
                const amrex::Real cjxx    = - jxx(i,j,k) / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cjxy    = - jxy(i,j,k) / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cjyy    = - jyy(i,j,k) / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cdx_jxx = - dx_jxx / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cdx_jxy = - dx_jxy / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cdx_jz  = - dx_jz  / n0 / pc.q_e / pc.c;
                const amrex::Real cdx_psi =   dx_psi * pc.q_e / (pc.m_e * pc.c * pc.c);
                const amrex::Real cdy_jyy = - dy_jyy / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cdy_jxy = - dy_jxy / n0 / pc.q_e / pc.c / pc.c;
                const amrex::Real cdy_jz  = - dy_jz  / n0 / pc.q_e / pc.c ;
                const amrex::Real cdy_psi =   dy_psi * pc.q_e / (pc.m_e * pc.c * pc.c);
                const amrex::Real cdz_jxb = - dz_jxb / n0 / pc.q_e / pc.c;
                const amrex::Real cdz_jyb = - dz_jyb / n0 / pc.q_e / pc.c;
                const amrex::Real cez     =   ez(i,j,k) / E0;
                const amrex::Real cbz     =   bz(i,j,k) * pc.c / E0;

                // to calculate nstar, only the plasma current density is needed
                const amrex::Real nstar = cne - cjzp;

                const amrex::Real nstar_gamma = 0.5_rt* (1._rt+cpsi)*(cjxx + cjyy + nstar)
                                                + 0.5_rt * nstar/(1._rt+cpsi);

                const amrex::Real nstar_ax = 1._rt/(1._rt + cpsi) *
                    (nstar_gamma*cdx_psi/(1._rt+cpsi) - cjxp*cez - cjxx*cdx_psi - cjxy*cdy_psi);

                const amrex::Real nstar_ay = 1._rt/(1._rt + cpsi) *
                    (nstar_gamma*cdy_psi/(1._rt+cpsi) - cjyp*cez - cjxy*cdx_psi - cjyy*cdy_psi);

                // Should only have 1 component, but not supported yet by the AMReX MG solver
                mult(i,j,k,0) = nstar / (1._rt + cpsi);
                mult(i,j,k,1) = nstar / (1._rt + cpsi);

                // sy, to compute Bx
                s(i,j,k,0) = + cbz * cjxp / (1._rt+cpsi) + nstar_ay - cdx_jxy - cdy_jyy + cdy_jz
                             + cdz_jyb;
                // sx, to compute By
                s(i,j,k,1) = - cbz * cjyp / (1._rt+cpsi) + nstar_ax - cdx_jxx - cdy_jxy + cdx_jz
                             + cdz_jxb;
                s(i,j,k,1) *= -1;

            }
            );
    }

#ifdef AMREX_USE_LINEAR_SOLVERS
    // For now, we construct the solver locally. Later, we want to move it to the hipace class as
    // a member so that we can reuse it.

    // construct slice geometry in normalized units
    // Set the lo and hi of domain and probdomain in the z direction
    amrex::RealBox tmp_probdom({AMREX_D_DECL(Geom(lev).ProbLo(Direction::x) / kpinv,
                                             Geom(lev).ProbLo(Direction::y) / kpinv,
                                             Geom(lev).ProbLo(Direction::z) / kpinv)},
                               {AMREX_D_DECL(Geom(lev).ProbHi(Direction::x) / kpinv,
                                             Geom(lev).ProbHi(Direction::y) / kpinv,
                                             Geom(lev).ProbHi(Direction::z) / kpinv)});
    amrex::Box tmp_dom = Geom(lev).Domain();
    const amrex::Real hi = Geom(lev).ProbHi(Direction::z) / kpinv;
    const amrex::Real lo = hi - dz;
    tmp_probdom.setLo(Direction::z, lo);
    tmp_probdom.setHi(Direction::z, hi);
    tmp_dom.setSmall(Direction::z, 0);
    tmp_dom.setBig(Direction::z, 0);
    amrex::Geometry slice_geom = amrex::Geometry(
        tmp_dom, tmp_probdom, Geom(lev).Coord(), Geom(lev).isPeriodic());

    slice_geom.setPeriodicity({0,0,0});

    if (!m_mlalaplacian){
        // If first call, initialize the MG solver
        amrex::LPInfo lpinfo{};
        lpinfo.setHiddenDirection(2).setAgglomeration(false).setConsolidation(false);

        // make_unique requires explicit types
        m_mlalaplacian = std::make_unique<amrex::MLALaplacian>(
            amrex::Vector<amrex::Geometry>{slice_geom},
            amrex::Vector<amrex::BoxArray>{S.boxArray()},
            amrex::Vector<amrex::DistributionMapping>{S.DistributionMap()},
            lpinfo,
            amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, 2);

        m_mlalaplacian->setDomainBC(
            {AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                          amrex::LinOpBCType::Dirichlet,
                          amrex::LinOpBCType::Dirichlet)},
            {AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                          amrex::LinOpBCType::Dirichlet,
                          amrex::LinOpBCType::Dirichlet)});

        m_mlmg = std::make_unique<amrex::MLMG>(*m_mlalaplacian);
    }

    // BxBy is assumed to have at least one ghost cell in x and y.
    // The ghost cells outside the domain should contain Dirichlet BC values.
    BxBy.setDomainBndry(0.0, slice_geom); // Set Dirichlet BC to zero
    m_mlalaplacian->setLevelBC(0, &BxBy);

    m_mlalaplacian->setACoeffs(0, Mult);

    // amrex solves ascalar A phi - bscalar Laplacian(phi) = rhs
    // So we solve Delta BxBy - A * BxBy = S
    m_mlalaplacian->setScalars(-1.0, -1.0);

    m_mlmg->solve({&BxBy}, {&S}, m_MG_tolerance_rel, m_MG_tolerance_abs);
#else
    amrex::Abort("To use the explicit solver, compilation option AMReX_LINEAR_SOLVERS must be ON");
#endif

    // converting BxBy to SI units, if applicable
    BxBy.mult(E0/pc.c);
    amrex::ParallelContext::pop();
}

void
Hipace::PredictorCorrectorLoopToSolveBxBy (const int islice, const int lev, const amrex::Box bx,
                        amrex::Vector<BeamBins> bins,
                        const int ibox)
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
    amrex::MultiFab jx_beam_next(m_fields.getSlices(lev, WhichSlice::Next),
                            amrex::make_alias, Comps[WhichSlice::Next]["jx_beam"], 1);
    amrex::MultiFab jy_beam_next(m_fields.getSlices(lev, WhichSlice::Next),
                            amrex::make_alias, Comps[WhichSlice::Next]["jy_beam"], 1);


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
            m_fields, WhichSlice::Next, true, true, false, false, false, geom[lev], lev);

        m_multi_beam.DepositCurrentSlice(m_fields, geom[lev], lev, islice, bx, bins, m_box_sorters,
                                         ibox, m_do_beam_jx_jy_deposition, WhichSlice::Next);
        m_fields.AddBeamCurrents(lev, WhichSlice::Next);

        amrex::ParallelContext::push(m_comm_xy);
        // need to exchange jx jy jx_beam jy_beam
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
        jx_beam_next.setVal(0.);
        jy_beam_next.setVal(0.);

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
Hipace::Wait (const int step, int it, bool only_ghost)
{
    HIPACE_PROFILE("Hipace::Wait()");

#ifdef AMREX_USE_MPI
    if (step == 0) return;

    // Receive physical time
    if (it == m_numprocs_z - 1 && !only_ghost) {
        MPI_Status status;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0
        MPI_Recv(&m_physical_time, 1,
                 amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 (m_rank_z+1)%m_numprocs_z, tcomm_z_tag, m_comm_z, &status);
    }

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
        const int loc_ncomm_z_tag = only_ghost ? ncomm_z_tag_ghost : ncomm_z_tag;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0
        MPI_Recv(np_rcv.dataPtr(), nint,
                 amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
                 (m_rank_z+1)%m_numprocs_z, loc_ncomm_z_tag, m_comm_z, &status);
    }
    if (!only_ghost) m_leftmost_box_rcv = std::min(np_rcv[nbeams], m_leftmost_box_rcv);

    // Receive beam particles.
    {
        const amrex::Long np_total = std::accumulate(np_rcv.begin(), np_rcv.begin()+nbeams, 0);
        if (np_total == 0) return;
        const amrex::Long psize = sizeof(BeamParticleContainer::SuperParticleType);
        const amrex::Long buffer_size = psize*np_total;
        auto recv_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

        MPI_Status status;
        const int loc_pcomm_z_tag = only_ghost ? pcomm_z_tag_ghost : pcomm_z_tag;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0
        MPI_Recv(recv_buffer, buffer_size,
                 amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                 (m_rank_z+1)%m_numprocs_z, loc_pcomm_z_tag, m_comm_z, &status);

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
Hipace::Notify (const int step, const int it,
                amrex::Vector<BeamBins>& bins, bool only_ghost)
{
    HIPACE_PROFILE("Hipace::Notify()");

    constexpr int lev = 0;

#ifdef AMREX_USE_MPI
    NotifyFinish(it, only_ghost); // finish the previous send

    const int nbeams = m_multi_beam.get_nbeams();
    const int nint = nbeams + 1;

    // last step does not need to send anything, but needs to resize to remove slipped particles
    if (step == m_max_step)
    {
        if (!only_ghost) {
            for (int ibeam = 0; ibeam < nbeams; ibeam++){
                const int offset_box = m_box_sorters[ibeam].boxOffsetsPtr()[it];
                auto& ptile = m_multi_beam.getBeam(ibeam);
                ptile.resize(offset_box);
            }
        }
        return;
    }

    // send physical time
    if (it == m_numprocs_z - 1 && !only_ghost){
        const amrex::Real t = m_physical_time + m_dt;
        MPI_Isend(&t, 1, amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                  (m_rank_z-1+m_numprocs_z)%m_numprocs_z, tcomm_z_tag, m_comm_z, &m_tsend_request);
    }

    m_leftmost_box_snd = std::min(m_leftmost_box_snd, m_leftmost_box_rcv);
    if (it < m_leftmost_box_snd && it < m_numprocs_z - 1 && m_skip_empty_comms){
        if (m_verbose >= 2){
            amrex::AllPrint()<<"rank "<<m_rank_z<<" step "<<step<<" box "<<it<<": SKIP SEND!\n";
        }
        return;
    }

    // 1 element per beam species, and 1 for the index of leftmost box with beam particles.
    amrex::Vector<int>& np_snd = only_ghost ? m_np_snd_ghost : m_np_snd;
    np_snd.resize(nint);

    const amrex::Box& bx = boxArray(lev)[it];
    for (int ibeam = 0; ibeam < nbeams; ++ibeam)
    {
        np_snd[ibeam] = only_ghost ?
            m_multi_beam.NGhostParticles(ibeam, bins, bx)
            : m_box_sorters[ibeam].boxCountsPtr()[it];
    }
    np_snd[nbeams] = m_leftmost_box_snd;

    // Each rank sends data downstream, except rank 0 who sends data to m_numprocs_z-1
    const int loc_ncomm_z_tag = only_ghost ? ncomm_z_tag_ghost : ncomm_z_tag;
    MPI_Request* loc_nsend_request = only_ghost ? &m_nsend_request_ghost : &m_nsend_request;
    MPI_Isend(np_snd.dataPtr(), nint, amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
              (m_rank_z-1+m_numprocs_z)%m_numprocs_z, loc_ncomm_z_tag, m_comm_z, loc_nsend_request);

    // Send beam particles. Currently only one tile.
    {
        const amrex::Long np_total = std::accumulate(np_snd.begin(), np_snd.begin()+nbeams, 0);
        if (np_total == 0) return;
        const amrex::Long psize = sizeof(BeamParticleContainer::SuperParticleType);
        const amrex::Long buffer_size = psize*np_total;
        char*& psend_buffer = only_ghost ? m_psend_buffer_ghost : m_psend_buffer;
        psend_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

        int offset_beam = 0;
        for (int ibeam = 0; ibeam < nbeams; ibeam++){
            const int offset_box = m_box_sorters[ibeam].boxOffsetsPtr()[it];
            const amrex::Long np = np_snd[ibeam];

            auto& ptile = m_multi_beam.getBeam(ibeam);
            const auto ptd = ptile.getConstParticleTileData();

            const amrex::Gpu::DeviceVector<int> comm_real(m_multi_beam.NumRealComps(), 1);
            const amrex::Gpu::DeviceVector<int> comm_int (m_multi_beam.NumIntComps(),  1);
            const auto p_comm_real = comm_real.data();
            const auto p_comm_int = comm_int.data();
            const auto p_psend_buffer = psend_buffer + offset_beam*psize;

            BeamBins::index_type* indices = nullptr;
            BeamBins::index_type const * offsets = 0;
            BeamBins::index_type cell_start = 0;

            indices = bins[ibeam].permutationPtr();
            offsets = bins[ibeam].offsetsPtr();

            // The particles that are in the last slice (sent as ghost particles) are
            // given by the indices[cell_start:cell_stop-1]
            cell_start = offsets[bx.bigEnd(Direction::z)-bx.smallEnd(Direction::z)];

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
                            const int src_i = only_ghost ? indices[cell_start+i] : i;
                            ptd.packParticleData(shared, offset_box+src_i, m*psize, p_comm_real, p_comm_int);
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
                    const int src_i = only_ghost ? indices[cell_start+i] : i;
                    ptd.packParticleData(p_psend_buffer, offset_box+src_i, i*psize, p_comm_real, p_comm_int);
                }
            }
            amrex::Gpu::Device::synchronize();

            // Delete beam particles that we just sent from the particle array
            if (!only_ghost) ptile.resize(offset_box);
            offset_beam += np;
        } // here

        const int loc_pcomm_z_tag = only_ghost ? pcomm_z_tag_ghost : pcomm_z_tag;
        MPI_Request* loc_psend_request = only_ghost ? &m_psend_request_ghost : &m_psend_request;
        // Each rank sends data downstream, except rank 0 who sends data to m_numprocs_z-1
        MPI_Isend(psend_buffer, buffer_size, amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                  (m_rank_z-1+m_numprocs_z)%m_numprocs_z, loc_pcomm_z_tag, m_comm_z, loc_psend_request);
    }
#endif
}

void
Hipace::NotifyFinish (const int it, bool only_ghost)
{
#ifdef AMREX_USE_MPI
    if (only_ghost) {
        if (m_np_snd_ghost.size() > 0) {
            MPI_Status status;
            MPI_Wait(&m_nsend_request_ghost, &status);
            m_np_snd_ghost.resize(0);
        }
        if (m_psend_buffer_ghost) {
            MPI_Status status;
            MPI_Wait(&m_psend_request_ghost, &status);
            amrex::The_Pinned_Arena()->free(m_psend_buffer_ghost);
            m_psend_buffer_ghost = nullptr;
        }
    } else {
        if (it == m_numprocs_z - 1) {
            MPI_Status status;
            MPI_Wait(&m_tsend_request, &status);
        }

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
    }
#endif
}

void
Hipace::FillDiagnostics (int lev, int i_slice)
{
    m_fields.Copy(lev, i_slice, FieldCopyType::StoF, 0, 0, Comps[WhichSlice::This]["N"],
                  m_diags.getF(lev), m_diags.sliceDir(), Geom(lev));
}

void
Hipace::WriteDiagnostics (int output_step, const int it, const OpenPMDWriterCallType call_type)
{
    HIPACE_PROFILE("Hipace::WriteDiagnostics()");

    // Dump every m_output_period steps and after last step
    if (m_output_period < 0 ||
        (!(output_step == m_max_step) && output_step % m_output_period != 0) ) return;

    // assumption: same order as in struct enum Field Comps
    const amrex::Vector< std::string > varnames = getDiagComps();
    const amrex::Vector< std::string > beamnames = getDiagBeamNames();

#ifdef HIPACE_USE_OPENPMD
    constexpr int lev = 0;
    m_openpmd_writer.WriteDiagnostics(getDiagF(), m_multi_beam, getDiagGeom(),
                        m_physical_time, output_step, lev, getDiagSliceDir(), varnames, beamnames,
                        it, m_box_sorters, geom[lev], call_type);
#else
    amrex::Print()<<"WARNING: HiPACE++ compiled without openPMD support, the simulation has no I/O.\n";
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

void
Hipace::CheckGhostSlice (int it)
{
    HIPACE_PROFILE("Hipace::CheckGhostSlice()");

    constexpr int lev = 0;

    if (it == 0) return;

    for (int ibeam=0; ibeam<m_multi_beam.get_nbeams(); ibeam++) {
        const int nreal = m_multi_beam.getNRealParticles(ibeam);
        const int nghost = m_multi_beam.Npart(ibeam) - nreal;

        if (m_verbose >= 3) {
            amrex::AllPrint()<<"CheckGhostSlice rank "<<m_rank_z<<" it "<<it
                             <<" npart "<<m_multi_beam.Npart(ibeam)<<" nreal "
                             <<nreal<<" nghost "<<nghost<<"\n";
        }

        // Get lo and hi indices of current box
        const amrex::Box& bx = boxArray(lev)[it];
        const int ilo = bx.smallEnd(Direction::z);

        // Get domain size in physical space
        const amrex::Real dz = Geom(lev).CellSize(Direction::z);
        const amrex::Real dom_lo = Geom(lev).ProbLo(Direction::z);

        // Compute bounds of ghost cell
        const amrex::Real zmin_leftcell = dom_lo + dz*(ilo-1);
        const amrex::Real zmax_leftcell = dom_lo + dz*ilo;

        // Get pointers to ghost particles
        auto& ptile = m_multi_beam.getBeam(ibeam);
        auto& aos = ptile.GetArrayOfStructs();
        const auto& pos_structs = aos.begin() + nreal;

        // Invalidate particles out of the ghost slice
        amrex::ParallelFor(
            nghost,
            [=] AMREX_GPU_DEVICE (long idx) {
                // Get zp of ghost particle
                const amrex::Real zp = pos_structs[idx].pos(2);
                // Invalidate ghost particle if not in the ghost slice
                if ( zp < zmin_leftcell || zp > zmax_leftcell ) {
                    pos_structs[idx].id() = -1;
                }
            }
            );
    }
}
