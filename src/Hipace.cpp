/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Remi Lehe, Severin Diederichs, WeiqunZhang, coulibaly-mouhamed
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "particles/sorting/SliceSort.H"
#include "particles/sorting/BoxSort.H"
#include "salame/Salame.H"
#include "utils/IOUtil.H"
#include "utils/GPUUtil.H"
#include "particles/pusher/GetAndSetPosition.H"
#include "mg_solver/HpMultiGrid.H"

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
    constexpr int lcomm_z_tag = 1006;
}
#endif

Hipace* Hipace::m_instance = nullptr;

bool Hipace::m_normalized_units = false;
int Hipace::m_max_step = 0;
amrex::Real Hipace::m_dt = 0.0;
amrex::Real Hipace::m_max_time = std::numeric_limits<amrex::Real>::infinity();
amrex::Real Hipace::m_physical_time = 0.0;
amrex::Real Hipace::m_initial_time = 0.0;
int Hipace::m_verbose = 0;
int Hipace::m_depos_order_xy = 2;
int Hipace::m_depos_order_z = 0;
int Hipace::m_depos_derivative_type = 2;
bool Hipace::m_outer_depos_loop = false;
amrex::Real Hipace::m_predcorr_B_error_tolerance = 4e-2;
int Hipace::m_predcorr_max_iterations = 30;
amrex::Real Hipace::m_predcorr_B_mixing_factor = 0.05;
bool Hipace::m_do_beam_jx_jy_deposition = true;
bool Hipace::m_do_beam_jz_minus_rho = false;
int Hipace::m_beam_injection_cr = 1;
amrex::Real Hipace::m_external_ExmBy_slope = 0.;
amrex::Real Hipace::m_external_Ez_slope = 0.;
amrex::Real Hipace::m_external_Ez_uniform = 0.;
amrex::Real Hipace::m_MG_tolerance_rel = 1.e-4;
amrex::Real Hipace::m_MG_tolerance_abs = 0.;
int Hipace::m_MG_verbose = 0;
bool Hipace::m_use_amrex_mlmg = false;
bool Hipace::m_use_laser = false;
bool Hipace::m_do_MR = false;

#ifdef AMREX_USE_GPU
bool Hipace::m_do_tiling = false;
#else
bool Hipace::m_do_tiling = true;
#endif

Hipace_early_init::Hipace_early_init (Hipace* instance)
{
    Hipace::m_instance = instance;
    amrex::ParmParse pph("hipace");
    queryWithParser(pph ,"normalized_units", Hipace::m_normalized_units);
    if (Hipace::m_normalized_units) {
        m_phys_const = make_constants_normalized();
    } else {
        m_phys_const = make_constants_SI();
    }
    Parser::addConstantsToParser(m_phys_const);
    Parser::replaceAmrexParamsWithParser();
}

Hipace&
Hipace::GetInstance ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_instance, "instance has not been initialized yet");
    return *m_instance;
}

Hipace::Hipace () :
    Hipace_early_init(this),
    amrex::AmrCore(),
    m_fields(this),
    m_multi_beam(this),
    m_multi_plasma(this),
    m_adaptive_time_step(m_multi_beam.get_nbeams()),
    m_multi_laser(),
    m_diags(this->maxLevel()+1)
{
    amrex::ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
    queryWithParser(pp, "max_step", m_max_step);

    int seed;
    if (queryWithParser(pp, "random_seed", seed)) amrex::ResetRandomSeed(seed);

    amrex::ParmParse pph("hipace");

    std::string str_dt {""};
    queryWithParser(pph, "dt", str_dt);
    if (str_dt != "adaptive") queryWithParser(pph, "dt", m_dt);
    queryWithParser(pph, "max_time", m_max_time);
    queryWithParser(pph, "verbose", m_verbose);
    queryWithParser(pph, "numprocs_x", m_numprocs_x);
    queryWithParser(pph, "numprocs_y", m_numprocs_y);
    m_numprocs_z = amrex::ParallelDescriptor::NProcs() / (m_numprocs_x*m_numprocs_y);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_z <= m_max_step+1,
                                     "Please use more or equal time steps than number of ranks");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_x*m_numprocs_y*m_numprocs_z
                                     == amrex::ParallelDescriptor::NProcs(),
                                     "Check hipace.numprocs_x and hipace.numprocs_y");
    queryWithParser(pph, "boxes_in_z", m_boxes_in_z);
    if (m_boxes_in_z > 1) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_numprocs_z == 1,
                            "Multiple boxes per rank only implemented for one rank.");
    queryWithParser(pph, "depos_order_xy", m_depos_order_xy);
    queryWithParser(pph, "depos_order_z", m_depos_order_z);
    queryWithParser(pph, "depos_derivative_type", m_depos_derivative_type);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_depos_order_xy != 0 || m_depos_derivative_type != 0,
                            "Analytic derivative with depos_order=0 would vanish");
    queryWithParser(pph, "outer_depos_loop", m_outer_depos_loop);
    queryWithParser(pph, "predcorr_B_error_tolerance", m_predcorr_B_error_tolerance);
    queryWithParser(pph, "predcorr_max_iterations", m_predcorr_max_iterations);
    queryWithParser(pph, "predcorr_B_mixing_factor", m_predcorr_B_mixing_factor);
    queryWithParser(pph, "output_period", m_output_period);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_output_period != 0,
                                     "To avoid output, please use output_period = -1.");
    queryWithParser(pph, "beam_injection_cr", m_beam_injection_cr);
    queryWithParser(pph, "do_beam_jx_jy_deposition", m_do_beam_jx_jy_deposition);
    queryWithParser(pph, "do_beam_jz_minus_rho", m_do_beam_jz_minus_rho);
    queryWithParser(pph, "do_device_synchronize", DO_DEVICE_SYNCHRONIZE);
    bool do_mfi_sync = false;
    queryWithParser(pph, "do_MFIter_synchronize", do_mfi_sync);
    DfltMfi.SetDeviceSync(do_mfi_sync).UseDefaultStream();
    DfltMfiTlng.SetDeviceSync(do_mfi_sync).UseDefaultStream();
    if (amrex::TilingIfNotGPU()) {
        DfltMfiTlng.EnableTiling();
    }
    queryWithParser(pph, "external_ExmBy_slope", m_external_ExmBy_slope);
    queryWithParser(pph, "external_Ez_slope", m_external_Ez_slope);
    queryWithParser(pph, "external_Ez_uniform", m_external_Ez_uniform);
    queryWithParser(pph, "salame_n_iter", m_salame_n_iter);
    queryWithParser(pph, "salame_do_advance", m_salame_do_advance);
    std::string salame_target_str = "Ez_initial";
    queryWithParser(pph, "salame_Ez_target(zeta,zeta_initial,Ez_initial)", salame_target_str);
    m_salame_target_func = makeFunctionWithParser<3>(salame_target_str, m_salame_parser,
                                                     {"zeta", "zeta_initial", "Ez_initial"});

    std::string solver = "predictor-corrector";
    queryWithParser(pph, "bxby_solver", solver);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solver == "predictor-corrector" ||
        solver == "explicit",
        "hipace.bxby_solver must be predictor-corrector or explicit");
    if (solver == "explicit") m_explicit = true;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_explicit || !m_multi_beam.AnySpeciesSalame(),
        "Cannot use SALAME algorithm with predictor-corrector solver");
    queryWithParser(pph, "MG_tolerance_rel", m_MG_tolerance_rel);
    queryWithParser(pph, "MG_tolerance_abs", m_MG_tolerance_abs);
    queryWithParser(pph, "MG_verbose", m_MG_verbose);
    queryWithParser(pph, "use_amrex_mlmg", m_use_amrex_mlmg);
    queryWithParser(pph, "do_tiling", m_do_tiling);
#ifdef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_do_tiling==0, "Tiling must be turned off to run on GPU.");
#endif
    m_do_MR = maxLevel() > 0;
    if (m_do_MR) {
        AMREX_ALWAYS_ASSERT(maxLevel() < 2);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        getWithParser(pph, "patch_lo", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) patch_lo[idim] = loc_array[idim];
        getWithParser(pph, "patch_hi", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) patch_hi[idim] = loc_array[idim];
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_do_MR || !m_multi_beam.AnySpeciesSalame(),
        "Cannot use SALAME algorithm with mesh refinement");

#ifdef AMREX_USE_MPI
    queryWithParser(pph, "skip_empty_comms", m_skip_empty_comms);
    int myproc = amrex::ParallelDescriptor::MyProc();
    m_rank_z = myproc/(m_numprocs_x*m_numprocs_y);
    MPI_Comm_split(amrex::ParallelDescriptor::Communicator(), m_rank_z, myproc, &m_comm_xy);
    MPI_Comm_rank(m_comm_xy, &m_rank_xy);
    MPI_Comm_split(amrex::ParallelDescriptor::Communicator(), m_rank_xy, myproc, &m_comm_z);
#endif

    m_use_laser = m_multi_laser.m_use_laser;
}

Hipace::~Hipace ()
{
#ifdef AMREX_USE_MPI
    if (m_physical_time < m_max_time) {
        NotifyFinish();
        NotifyFinish(0, true);
    } else {
        NotifyFinish(0, false, true); // finish only time sends
    }
    MPI_Comm_free(&m_comm_xy);
    MPI_Comm_free(&m_comm_z);
#endif
}

void
Hipace::DefineSliceGDB (const int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
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
    m_slice_ba.push_back(amrex::BoxArray(std::move(bl)));

    // Slice DistributionMapping
    m_slice_dm.push_back(amrex::DistributionMapping(std::move(procmap)));

    // Slice Geometry
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
    m_slice_geom.push_back(amrex::Geometry(
        tmp_dom, tmp_probdom, Geom(lev).Coord(), Geom(lev).isPeriodic()));
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
#ifdef AMREX_USE_FLOAT
    amrex::Print() << "HiPACE++ (" << Hipace::Version() << ") running in single precision\n";
#else
    amrex::Print() << "HiPACE++ (" << Hipace::Version() << ") running in double precision\n";
#endif
#ifdef AMREX_USE_CUDA
    amrex::Print() << "using CUDA version " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__
                   << "." << __CUDACC_VER_BUILD__ << "\n";
#endif


    amrex::Vector<amrex::IntVect> new_max_grid_size;
    for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
        amrex::IntVect mgs = maxGridSize(ilev);
        mgs[0] = mgs[1] = 1024000000; // disable domain decomposition in x and y directions
        mgs[2] = Geom(ilev).Domain().length()[2]/m_numprocs_z; // make 1 box per rank longitudinally
        new_max_grid_size.push_back(mgs);
    }
    SetMaxGridSize(new_max_grid_size);

    AmrCore::InitFromScratch(0.0); // function argument is time
    constexpr int lev = 0;
    m_initial_time = m_multi_beam.InitData(geom[lev]);
    m_multi_plasma.InitData(m_slice_ba, m_slice_dm, m_slice_geom, geom);
    m_adaptive_time_step.Calculate(m_dt, m_multi_beam, m_multi_plasma.maxDensity());
#ifdef AMREX_USE_MPI
    m_adaptive_time_step.WaitTimeStep(m_dt, m_comm_z);
    m_adaptive_time_step.NotifyTimeStep(m_dt, m_comm_z);
#endif
    m_physical_time = m_initial_time;

    m_fields.checkInit();
}

void
Hipace::MakeNewLevelFromScratch (
    int lev, amrex::Real /*time*/, const amrex::BoxArray& ba, const amrex::DistributionMapping&)
{

    // We are going to ignore the DistributionMapping argument and build our own.
    amrex::DistributionMapping dm;
    {
        const int nboxes_x = m_numprocs_x;
        const int nboxes_y = m_numprocs_y;
        const int nboxes_z = (m_boxes_in_z == 1) ? m_numprocs_z : m_boxes_in_z;
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
    DefineSliceGDB(lev, ba, dm);
    m_fields.AllocData(lev, Geom(), m_slice_ba[lev], m_slice_dm[lev],
                       m_multi_plasma.m_sort_bin_size);
    m_multi_laser.InitData(m_slice_ba[0], m_slice_dm[0]); // laser inits only on level 0
    m_diags.Initialize(lev, m_multi_laser.m_use_laser);
}

void
Hipace::ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/)
{
    using namespace amrex::literals;
    const amrex::Real* problo = Geom(lev).ProbLo();
    const amrex::Real* dx = Geom(lev).CellSize();

    for (amrex::MFIter mfi(tags, DfltMfi); mfi.isValid(); ++mfi)
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
        if (m_physical_time <= m_max_time) {
            m_openpmd_writer.InitDiagnostics(step, m_output_period, m_max_step, finestLevel()+1);
        }
#endif

        ResetAllQuantities();

        // Loop over longitudinal boxes on this rank, from head to tail
        const int n_boxes = (m_boxes_in_z == 1) ? m_numprocs_z : m_boxes_in_z;
        for (int it = n_boxes-1; it >= 0; --it)
        {
            const amrex::Box& bx = boxArray(lev)[it];

            if (m_multi_laser.m_use_laser) {
                AMREX_ALWAYS_ASSERT(!m_adaptive_time_step.m_do_adaptive_time_step);
                AMREX_ALWAYS_ASSERT(m_multi_plasma.GetNPlasmas() <= 1);
                // Before that, the 3D fields of the envelope are not initialized (not even allocated).
                m_multi_laser.Init3DEnvelope(step, bx, Geom(0));
            }

            Wait(step, it);
            if (it == n_boxes-1) {
                // Only reset plasma after receiving time step, to use proper density
                // WARNING: handling of lev is to be improved: this loops over levels, but
                // lev is set to 0 above.
                for (int lv=0; lv<=finestLevel(); ++lv) m_multi_plasma.ResetParticles(lv, true);
                /* Store charge density of (immobile) ions into WhichSlice::RhoIons */
                if (m_do_tiling) m_multi_plasma.TileSort(boxArray(lev)[0], geom[lev]);
                m_multi_plasma.DepositNeutralizingBackground(
                    m_fields, m_multi_laser, WhichSlice::RhoIons, geom[lev], finestLevel()+1);
            }

            if (m_physical_time >= m_max_time) {
                Notify(step, it); // just send signal to finish simulation
                if (m_physical_time > m_max_time) break;
            }
            // adjust time step to reach max_time
            m_dt = std::min(m_dt, m_max_time - m_physical_time);

#ifdef HIPACE_USE_OPENPMD
            if (m_physical_time == m_max_time && it == n_boxes-1) { // init diagnostic if max_time
                m_openpmd_writer.InitDiagnostics(step, m_output_period, step, finestLevel()+1);
            }
#endif

            if (m_verbose>=1 && it==n_boxes-1) std::cout<<"Rank "<<rank<<" started  step "<<step
                                    <<" at time = "<<m_physical_time<< " with dt = "<<m_dt<<'\n';

            m_box_sorters.clear();

            m_multi_beam.sortParticlesByBox(m_box_sorters, boxArray(lev), geom[lev]);
            m_leftmost_box_snd = std::min(leftmostBoxWithParticles(), m_leftmost_box_snd);

            WriteDiagnostics(step, it, OpenPMDWriterCallType::beams);

            m_multi_beam.StoreNRealParticles();
            // Copy particles in box it-1 in the ghost buffer.
            // This handles both beam initialization and particle slippage.
            if (it>0) m_multi_beam.PackLocalGhostParticles(it-1, m_box_sorters);

            ResizeFDiagFAB(it);

            amrex::Vector<amrex::Vector<BeamBins>> bins;
            bins = m_multi_beam.findParticlesInEachSlice(finestLevel()+1, it, bx,
                                                         geom, m_box_sorters);
            AMREX_ALWAYS_ASSERT( bx.bigEnd(Direction::z) >= bx.smallEnd(Direction::z) + 2 );
            // Solve head slice
            SolveOneSlice(bx.bigEnd(Direction::z), it, step, bins);
            // Notify ghost slice
            if (it<m_numprocs_z-1) Notify(step, it, bins[lev], true);
            // Solve central slices
            for (int isl = bx.bigEnd(Direction::z)-1; isl > bx.smallEnd(Direction::z); --isl){
                SolveOneSlice(isl, it, step, bins);
            };
            // Receive ghost slice
            if (it>0) Wait(step, it, true);
            CheckGhostSlice(it);
            // Solve tail slice. Consume ghost particles.
            SolveOneSlice(bx.smallEnd(Direction::z), it, step, bins);
            // Delete ghost particles
            m_multi_beam.RemoveGhosts();

            if (m_physical_time < m_max_time) {
                m_adaptive_time_step.Calculate(m_dt, m_multi_beam, m_multi_plasma.maxDensity(),
                                               it, m_box_sorters, false);
            } else {
                m_dt = 2.*m_max_time;
            }


            // averaging predictor corrector loop diagnostics
            m_predcorr_avg_iterations /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));
            m_predcorr_avg_B_error /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));

            WriteDiagnostics(step, it, OpenPMDWriterCallType::fields);
            Notify(step, it, bins[lev]);
        }

        m_multi_beam.InSituWriteToFile(step, m_physical_time, geom[lev]);

        // printing and resetting predictor corrector loop diagnostics
        if (m_verbose>=2 && !m_explicit) amrex::AllPrint() << "Rank " << rank <<
                                ": avg. number of iterations " << m_predcorr_avg_iterations <<
                                " avg. transverse B field error " << m_predcorr_avg_B_error << "\n";
        m_predcorr_avg_iterations = 0.;
        m_predcorr_avg_B_error = 0.;

        m_physical_time += m_dt;

#ifdef HIPACE_USE_OPENPMD
        m_openpmd_writer.reset(step);
#endif
    }
}

void
Hipace::SolveOneSlice (int islice_coarse, const int ibox, int step,
                       const amrex::Vector<amrex::Vector<BeamBins>>& bins)
{
    HIPACE_PROFILE("Hipace::SolveOneSlice()");

    m_multi_beam.InSituComputeDiags(step, islice_coarse, bins[0],
                                    boxArray(0)[ibox].smallEnd(Direction::z),
                                    m_box_sorters, ibox);
    // Get this laser slice from the 3D array
    m_multi_laser.Copy(islice_coarse, false);

    for (int lev = 0; lev <= finestLevel(); ++lev) {

        if (lev == 1) { // skip all slices which are not existing on level 1
            // use geometry of coarse grid to determine whether slice is to be solved
            const amrex::Real* problo = Geom(0).ProbLo();
            const amrex::Real* dx = Geom(0).CellSize();
            amrex::Real pos = (islice_coarse+0.5)*dx[2]+problo[2];
            if (pos < patch_lo[2] || pos > patch_hi[2]) continue;
        }

        // Between this push and the corresponding pop at the end of this
        // for loop, the parallelcontext is the transverse communicator
        amrex::ParallelContext::push(m_comm_xy);

        const amrex::Box& bx = boxArray(lev)[ibox];

        const int nsubslice = GetRefRatio(lev)[Direction::z];

        for (int isubslice = nsubslice-1; isubslice >= 0; --isubslice) {

            // calculate correct slice for refined level
            const int islice = nsubslice*islice_coarse + isubslice;
            const int islice_local = islice - bx.smallEnd(Direction::z);

            if (m_explicit) {
                ExplicitSolveOneSubSlice(lev, step, ibox, bx, islice, islice_local, bins[lev]);
            } else {
                PredictorCorrectorSolveOneSubSlice(lev, step, ibox, bx, islice,
                                                   islice_local, bins[lev]);
            }

            FillDiagnostics(lev, islice);

            m_multi_plasma.doCoulombCollision(lev, bx, geom[lev]);

            m_multi_plasma.DoFieldIonization(lev, geom[lev], m_fields);

            if (m_multi_plasma.IonizationOn() && m_do_tiling) m_multi_plasma.TileSort(bx, geom[lev]);

            if (lev == 0) {
                // Advance laser slice by 1 step and store result to 3D array
                m_multi_laser.AdvanceSlice(m_fields, Geom(0), m_dt, step);
                m_multi_laser.Copy(islice_coarse, true);
            }

            if (lev != 0) {
                // shift slices of level 1
                m_fields.ShiftSlices(lev, islice_coarse, Geom(0), patch_lo[2], patch_hi[2]);
            }
        } // end for (int isubslice = nsubslice-1; isubslice >= 0; --isubslice)

        // After this, the parallel context is the full 3D communicator again
        amrex::ParallelContext::pop();

    } // end for (int lev = 0; lev <= finestLevel(); ++lev)

    // shift level 0
    m_fields.ShiftSlices(0, islice_coarse, Geom(0), patch_lo[2], patch_hi[2]);
}


void
Hipace::ExplicitSolveOneSubSlice (const int lev, const int step, const int ibox,
                                  const amrex::Box& bx, const int islice, const int islice_local,
                                  const amrex::Vector<BeamBins>& beam_bin)
{
    // Set all quantities to 0 except:
    // Bx and By: the previous slice serves as initial guess.
    // jx_beam and jy_beam are used from the previous "Next" slice
    // jx and jy are initially set to jx_beam and jy_beam
    m_fields.setVal(0., lev, WhichSlice::This, "chi", "Sy", "Sx", "ExmBy", "EypBx", "Ez",
        "Bz", "Psi", "jz_beam", "rho_beam", "jz", "rho");

    if (m_do_tiling) m_multi_plasma.TileSort(bx, geom[lev]);

    // deposit jx, jy, jz, rho and chi for all plasmas
    m_multi_plasma.DepositCurrent(
        m_fields, m_multi_laser, WhichSlice::This, false, true, true, true, true, geom[lev], lev);

    m_fields.setVal(0., lev, WhichSlice::Next, "jx_beam", "jy_beam");
    // deposit jx_beam and jy_beam in the Next slice
    m_multi_beam.DepositCurrentSlice(m_fields, geom, lev, step, islice_local, beam_bin,
        m_box_sorters, ibox, m_do_beam_jx_jy_deposition, false, false, WhichSlice::Next);
    // need to exchange jx_beam jy_beam
    m_fields.FillBoundary(Geom(lev).periodicity(), lev, WhichSlice::Next, "jx_beam", "jy_beam");

    m_fields.AddRhoIons(lev);

    // deposit jz_beam and maybe rho_beam on This slice
    m_multi_beam.DepositCurrentSlice(m_fields, geom, lev, step, islice_local, beam_bin,
        m_box_sorters, ibox, false, true, m_do_beam_jz_minus_rho, WhichSlice::This);

    FillBoundaryChargeCurrents(lev);

    m_fields.SolvePoissonExmByAndEypBx(Geom(), m_comm_xy, lev, islice);
    m_fields.SolvePoissonEz(Geom(), lev, islice);
    m_fields.SolvePoissonBz(Geom(), lev, islice);

    // deposit grid current into jz_beam
    m_grid_current.DepositCurrentSlice(m_fields, geom[lev], lev, islice);
    // No FillBoundary because grid current only deposits in the middle of the field

    // Set Sx and Sy to beam contribution
    InitializeSxSyWithBeam(lev);

    // Deposit Sx and Sy for every plasma species
    m_multi_plasma.ExplicitDeposition(m_fields, m_multi_laser, geom[lev], lev);

    // Solves Bx, By using Sx, Sy and chi
    ExplicitMGSolveBxBy(lev, WhichSlice::This, islice);

    const bool do_salame = m_multi_beam.isSalameNow(step, islice_local, beam_bin);
    if (do_salame) {
        // Modify the beam particle weights on this slice to flatten Ez.
        // As the beam current is modified, Bx and By are also recomputed.
        // Plasma particle force terms get shifted
        SalameModule(this, m_salame_n_iter, m_salame_do_advance, m_salame_last_slice,
                     m_salame_overloaded, lev, step, islice, islice_local, beam_bin, ibox);
    }

    // shift and update force terms, push plasma particles
    // don't shift force terms again if salame was used
    m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, geom[lev],
                                    false, true, true, !do_salame, lev);

    // Push beam particles
    m_multi_beam.AdvanceBeamParticlesSlice(m_fields, geom[lev], lev, islice_local, bx,
                                           beam_bin, m_box_sorters, ibox);
}

void
Hipace::PredictorCorrectorSolveOneSubSlice (const int lev, const int step, const int ibox,
                                            const amrex::Box& bx, const int islice,
                                            const int islice_local,
                                            const amrex::Vector<BeamBins>& beam_bin)
{
    m_fields.setVal(0., lev, WhichSlice::This,
        "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi");
    if (m_use_laser) {
        m_fields.setVal(0., lev, WhichSlice::This, "chi");
    }

    // push plasma
    m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, geom[lev], false,
                                    true, false, false, lev);

    if (m_do_tiling) m_multi_plasma.TileSort(bx, geom[lev]);
    // deposit jx jy jz rho and maybe chi
    m_multi_plasma.DepositCurrent(
        m_fields, m_multi_laser, WhichSlice::This, false, true, true, true, m_use_laser, geom[lev], lev);

    m_fields.AddRhoIons(lev);

    FillBoundaryChargeCurrents(lev);

    if (!m_do_beam_jz_minus_rho) {
        m_fields.SolvePoissonExmByAndEypBx(Geom(), m_comm_xy, lev, islice);
    }

    // deposit jx jy jz and maybe rho on This slice
    m_multi_beam.DepositCurrentSlice(m_fields, geom, lev, step, islice_local, beam_bin,
                                     m_box_sorters, ibox, m_do_beam_jx_jy_deposition, true,
                                     m_do_beam_jz_minus_rho, WhichSlice::This);

    if (m_do_beam_jz_minus_rho) {
        m_fields.SolvePoissonExmByAndEypBx(Geom(), m_comm_xy, lev, islice);
    }

    // deposit grid current into jz_beam
    m_grid_current.DepositCurrentSlice(m_fields, geom[lev], lev, islice);

    FillBoundaryChargeCurrents(lev);

    m_fields.SolvePoissonEz(Geom(), lev, islice);
    m_fields.SolvePoissonBz(Geom(), lev, islice);

    // Solves Bx and By in the current slice and modifies the force terms of the plasma particles
    PredictorCorrectorLoopToSolveBxBy(islice_local, lev, step, beam_bin, ibox);

    // Push beam particles
    m_multi_beam.AdvanceBeamParticlesSlice(m_fields, geom[lev], lev, islice_local, bx,
                                           beam_bin, m_box_sorters, ibox);
}

void
Hipace::ResetAllQuantities ()
{
    HIPACE_PROFILE("Hipace::ResetAllQuantities()");

    if (m_use_laser) ResetLaser();

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        for (amrex::MultiFab& slice : m_fields.getSlices(lev)) {
            if (slice.nComp() != 0) {
                slice.setVal(0., m_fields.m_slices_nguards);
            }
        }
    }
}

void
Hipace::ResetLaser ()
{
    HIPACE_PROFILE("Hipace::ResetLaser()");

    for (int sl=WhichLaserSlice::nm1j00; sl<WhichLaserSlice::N; sl++) {
        m_multi_laser.getSlices(sl).setVal(0.);
    }
}

void
Hipace::FillBoundaryChargeCurrents (int lev) {
    if (!m_fields.m_extended_solve) {
        if (m_explicit) {
            m_fields.FillBoundary(Geom(lev).periodicity(), lev, WhichSlice::This,
                "jx_beam", "jy_beam", "jz_beam", "rho_beam", "jx", "jy", "jz", "rho");
        } else {
            m_fields.FillBoundary(Geom(lev).periodicity(), lev, WhichSlice::This,
                "jx", "jy", "jz", "rho");
        }
    }
}

void
Hipace::InitializeSxSyWithBeam (const int lev)
{
    HIPACE_PROFILE("Hipace::InitializeSxSyWithBeam()");
    using namespace amrex::literals;

    amrex::MultiFab& slicemf = m_fields.getSlices(lev, WhichSlice::This);
    const amrex::MultiFab& nslicemf = m_fields.getSlices(lev, WhichSlice::Next);
    const amrex::MultiFab& pslicemf = m_fields.getSlices(lev, WhichSlice::Previous1);

    const amrex::Real dx = Geom(lev).CellSize(Direction::x);
    const amrex::Real dy = Geom(lev).CellSize(Direction::y);
    const amrex::Real dz = Geom(lev).CellSize(Direction::z);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        amrex::Box const& bx = mfi.tilebox();

        Array3<amrex::Real> const isl_arr = slicemf.array(mfi);
        Array3<const amrex::Real> const nsl_arr = nslicemf.const_array(mfi);
        Array3<const amrex::Real> const psl_arr = pslicemf.const_array(mfi);

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];
        const int next_jxb = Comps[WhichSlice::Next]["jx_beam"];
        const int next_jyb = Comps[WhichSlice::Next]["jy_beam"];
        const int      jzb = Comps[WhichSlice::This]["jz_beam"];
        const int prev_jxb = Comps[WhichSlice::Previous1]["jx_beam"];
        const int prev_jyb = Comps[WhichSlice::Previous1]["jy_beam"];

        const amrex::Real mu0 = m_phys_const.mu0;

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                const amrex::Real dx_jzb = (isl_arr(i+1,j,jzb)-isl_arr(i-1,j,jzb))/(2._rt*dx);
                const amrex::Real dy_jzb = (isl_arr(i,j+1,jzb)-isl_arr(i,j-1,jzb))/(2._rt*dy);
                const amrex::Real dz_jxb = (psl_arr(i,j,prev_jxb)-nsl_arr(i,j,next_jxb))/(2._rt*dz);
                const amrex::Real dz_jyb = (psl_arr(i,j,prev_jyb)-nsl_arr(i,j,next_jyb))/(2._rt*dz);

                // calculate contribution to Sx and Sy by all beams (same as with PC solver)
                // sy, to compute Bx
                isl_arr(i,j,Sy) =   mu0 * (
                                    - dy_jzb
                                    + dz_jyb);
                // sx, to compute By
                isl_arr(i,j,Sx) = - mu0 * (
                                    - dx_jzb
                                    + dz_jxb);
            });
    }
}


void
Hipace::ExplicitMGSolveBxBy (const int lev, const int which_slice, const int islice)
{
    HIPACE_PROFILE("Hipace::ExplicitMGSolveBxBy()");
    amrex::ParallelContext::push(m_comm_xy);

    // always get chi from WhichSlice::This
    const int which_slice_chi = WhichSlice::This;

    int ncomp_chi = 1;
#ifdef AMREX_USE_LINEAR_SOLVERS
    // 2 components only for AMReX MLMG
    if (m_use_amrex_mlmg) {
        ncomp_chi = 2;
        AMREX_ALWAYS_ASSERT(Comps[which_slice_chi]["chi"] + 1 == Comps[which_slice_chi]["chi2"]);
    }
#endif
    AMREX_ALWAYS_ASSERT(Comps[which_slice]["Bx"] + 1 == Comps[which_slice]["By"]);
    AMREX_ALWAYS_ASSERT(Comps[which_slice]["Sy"] + 1 == Comps[which_slice]["Sx"]);

    amrex::MultiFab& slicemf_BxBySySx = m_fields.getSlices(lev, which_slice);
    amrex::MultiFab BxBy (slicemf_BxBySySx, amrex::make_alias, Comps[which_slice]["Bx"], 2);
    amrex::MultiFab SySx (slicemf_BxBySySx, amrex::make_alias, Comps[which_slice]["Sy"], 2);

    amrex::MultiFab& slicemf_chi = m_fields.getSlices(lev, which_slice_chi);
    amrex::MultiFab Mult (slicemf_chi, amrex::make_alias, Comps[which_slice_chi]["chi"], ncomp_chi);

    if (lev!=0) {
        m_fields.SetBoundaryCondition(Geom(), lev, "Bx", islice,
                                      m_fields.getField(lev, which_slice, "Sy"));
        m_fields.SetBoundaryCondition(Geom(), lev, "By", islice,
                                      m_fields.getField(lev, which_slice, "Sx"));
    }

#ifdef AMREX_USE_LINEAR_SOLVERS
    if (m_use_amrex_mlmg) {
        // Copy chi to chi2
        m_fields.duplicate(lev, which_slice_chi, {"chi2"}, which_slice_chi, {"chi"});
        amrex::Gpu::streamSynchronize();
        if (m_mlalaplacian.size()<maxLevel()+1) {
            m_mlalaplacian.resize(maxLevel()+1);
            m_mlmg.resize(maxLevel()+1);
        }

        // construct slice geometry
        const amrex::RealBox slice_box{slicemf_BxBySySx.boxArray()[0], m_slice_geom[lev].CellSize(),
                                       m_slice_geom[lev].ProbLo()};
        amrex::Geometry slice_geom{slicemf_BxBySySx.boxArray()[0], slice_box,
                                   m_slice_geom[lev].CoordInt(), {0,0,0}};

        if (!m_mlalaplacian[lev]){
            // If first call, initialize the MG solver
            amrex::LPInfo lpinfo{};
            lpinfo.setHiddenDirection(2).setAgglomeration(false).setConsolidation(false);

            // make_unique requires explicit types
            m_mlalaplacian[lev] = std::make_unique<amrex::MLALaplacian>(
                amrex::Vector<amrex::Geometry>{slice_geom},
                amrex::Vector<amrex::BoxArray>{slicemf_BxBySySx.boxArray()},
                amrex::Vector<amrex::DistributionMapping>{slicemf_BxBySySx.DistributionMap()},
                lpinfo,
                amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, 2);

            m_mlalaplacian[lev]->setDomainBC(
                {AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                              amrex::LinOpBCType::Dirichlet,
                              amrex::LinOpBCType::Dirichlet)},
                {AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                              amrex::LinOpBCType::Dirichlet,
                              amrex::LinOpBCType::Dirichlet)});

            m_mlmg[lev] = std::make_unique<amrex::MLMG>(*(m_mlalaplacian[lev]));
            m_mlmg[lev]->setVerbose(m_MG_verbose);
        }

        // BxBy is assumed to have at least one ghost cell in x and y.
        // The ghost cells outside the domain should contain Dirichlet BC values.
        BxBy.setDomainBndry(0.0, slice_geom); // Set Dirichlet BC to zero
        m_mlalaplacian[lev]->setLevelBC(0, &BxBy);

        m_mlalaplacian[lev]->setACoeffs(0, Mult);

        // amrex solves ascalar A phi - bscalar Laplacian(phi) = rhs
        // So we solve Delta BxBy - A * BxBy = S
        m_mlalaplacian[lev]->setScalars(-1.0, -1.0);

        m_mlmg[lev]->solve({&BxBy}, {&SySx}, m_MG_tolerance_rel, m_MG_tolerance_abs);
    } else
#endif
    {
        AMREX_ALWAYS_ASSERT(slicemf_BxBySySx.boxArray().size() == 1);
        AMREX_ALWAYS_ASSERT(slicemf_chi.boxArray().size() == 1);
        if (m_hpmg.size()<maxLevel()+1) {
            m_hpmg.resize(maxLevel()+1);
        }
        if (!m_hpmg[lev]) {
            m_hpmg[lev] = std::make_unique<hpmg::MultiGrid>(m_slice_geom[lev].CellSize(0),
                                                            m_slice_geom[lev].CellSize(1),
                                                            slicemf_BxBySySx.boxArray()[0]);
        }
        const int max_iters = 200;
        m_hpmg[lev]->solve1(BxBy[0], SySx[0], Mult[0], m_MG_tolerance_rel, m_MG_tolerance_abs,
                            max_iters, m_MG_verbose);
    }
    amrex::ParallelContext::pop();
}

void
Hipace::PredictorCorrectorLoopToSolveBxBy (const int islice_local, const int lev, const int step,
                                           const amrex::Vector<BeamBins>& bins, const int ibox)
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

    if (!m_fields.m_extended_solve) {
        amrex::ParallelContext::push(m_comm_xy);
        // exchange ExmBy EypBx Ez Bx By Bz
        m_fields.FillBoundary(Geom(lev).periodicity(), lev, WhichSlice::This,
            "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi");
        amrex::ParallelContext::pop();
    }

    /* creating temporary Bx and By arrays for the current and previous iteration */
    amrex::MultiFab Bx_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                            m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                            m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab By_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                            m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                            m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    Bx_iter.setVal(0.0, m_fields.m_slices_nguards);
    By_iter.setVal(0.0, m_fields.m_slices_nguards);
    amrex::MultiFab Bx_prev_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                                 m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                                 m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab::Copy(Bx_prev_iter, m_fields.getSlices(lev, WhichSlice::This),
                          Comps[WhichSlice::This]["Bx"], 0, 1, m_fields.m_slices_nguards);
    amrex::MultiFab By_prev_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                                 m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                                 m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab::Copy(By_prev_iter, m_fields.getSlices(lev, WhichSlice::This),
                          Comps[WhichSlice::This]["By"], 0, 1, m_fields.m_slices_nguards);

    // shift force terms, update force terms using guessed Bx and By
    m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, geom[lev], false, false, true, true, lev);

    const int islice = islice_local + boxArray(lev)[ibox].smallEnd(Direction::z);

    // Begin of predictor corrector loop
    int i_iter = 0;
    // resetting the initial B-field error for mixing between iterations
    relative_Bfield_error = 1.0;
    while (( relative_Bfield_error > m_predcorr_B_error_tolerance )
           && ( i_iter < m_predcorr_max_iterations ))
    {
        i_iter++;
        m_predcorr_avg_iterations += 1.0;

        // Push particles to the next temp slice
        m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, geom[lev], true, true, false, false, lev);

        if (m_do_tiling) m_multi_plasma.TileSort(boxArray(lev)[0], geom[lev]);
        // plasmas deposit jx jy to next temp slice
        m_multi_plasma.DepositCurrent(
            m_fields, m_multi_laser, WhichSlice::Next, true, true, false, false, false, geom[lev], lev);

        // beams deposit jx jy to the next slice
        m_multi_beam.DepositCurrentSlice(m_fields, geom, lev, step, islice_local, bins,
            m_box_sorters, ibox, m_do_beam_jx_jy_deposition, false, false, WhichSlice::Next);

        if (!m_fields.m_extended_solve) {
            amrex::ParallelContext::push(m_comm_xy);
            // need to exchange jx jy jx_beam jy_beam
            m_fields.FillBoundary(Geom(lev).periodicity(), lev, WhichSlice::Next,
                "jx", "jy");
            amrex::ParallelContext::pop();
        }

        /* Calculate Bx and By */
        m_fields.SolvePoissonBx(Bx_iter, Geom(), lev, islice);
        m_fields.SolvePoissonBy(By_iter, Geom(), lev, islice);

        relative_Bfield_error = m_fields.ComputeRelBFieldError(
            m_fields.getSlices(lev, WhichSlice::This),
            m_fields.getSlices(lev, WhichSlice::This),
            Bx_iter, By_iter,
            Comps[WhichSlice::This]["Bx"], Comps[WhichSlice::This]["By"],
            0, 0, Geom(lev));

        if (i_iter == 1) relative_Bfield_error_prev_iter = relative_Bfield_error;

        // Mixing the calculated B fields to the actual B field and shifting iterated B fields
        m_fields.MixAndShiftBfields(
            Bx_iter, Bx_prev_iter, Comps[WhichSlice::This]["Bx"], relative_Bfield_error,
            relative_Bfield_error_prev_iter, m_predcorr_B_mixing_factor, lev);
        m_fields.MixAndShiftBfields(
            By_iter, By_prev_iter, Comps[WhichSlice::This]["By"], relative_Bfield_error,
            relative_Bfield_error_prev_iter, m_predcorr_B_mixing_factor, lev);

        // resetting current in the next slice to clean temporarily used current
        m_fields.setVal(0., lev, WhichSlice::Next, "jx", "jy");

        if (!m_fields.m_extended_solve) {
            amrex::ParallelContext::push(m_comm_xy);
            // exchange Bx By
            m_fields.FillBoundary(Geom(lev).periodicity(), lev, WhichSlice::This,
                "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi");
            amrex::ParallelContext::pop();
        }

        // Update force terms using the calculated Bx and By
        m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, geom[lev],
                                        false, false, true, false, lev);

        // Shift relative_Bfield_error values
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
    if (m_verbose >= 2) amrex::Print()<<"level: " << lev << " islice: " << islice <<
                " n_iter: "<<i_iter<<" relative B field error: "<<relative_Bfield_error<< "\n";
}

void
Hipace::Wait (const int step, int it, bool only_ghost)
{
    HIPACE_PROFILE("Hipace::Wait()");

#ifdef AMREX_USE_MPI
    if (step == 0) return;
    if (m_numprocs_z == 1) return;

    // Receive physical time
    if (it == m_numprocs_z - 1 && !only_ghost) {
        MPI_Status status;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0
        MPI_Recv(&m_physical_time, 1,
                 amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 (m_rank_z+1)%m_numprocs_z, tcomm_z_tag, m_comm_z, &status);
    }

    if (m_physical_time > m_max_time) return;

    const int nbeams = m_multi_beam.get_nbeams();
    // 1 element per beam species,
    // 1 for the index of leftmost box with beam particles,
    // 1 for number of z points for laser data.
    const int nint = nbeams + 2;
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
    const int nz_laser = np_rcv[nbeams+1];
    if (!only_ghost) m_leftmost_box_rcv = std::min(np_rcv[nbeams], m_leftmost_box_rcv);

    // Receive beam particles.
    {
        const amrex::Long np_total = std::accumulate(np_rcv.begin(), np_rcv.begin()+nbeams, 0);
        if (np_total == 0 && !m_multi_laser.m_use_laser) return;
        const amrex::Long psize = sizeof(BeamParticleContainer::SuperParticleType);
        const amrex::Long buffer_size = psize*np_total;
        auto recv_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

        MPI_Status status;
        const int loc_pcomm_z_tag = only_ghost ? pcomm_z_tag_ghost : pcomm_z_tag;
        // Each rank receives data from upstream, except rank m_numprocs_z-1 who receives from 0

        // Make datatype the same size as one particle, so MAX_INT particles can be sent
        MPI_Datatype one_particle_size{};
        MPI_Type_contiguous(psize, amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                            &one_particle_size);
        MPI_Type_commit(&one_particle_size);

        MPI_Recv(recv_buffer, np_total, one_particle_size,
                 (m_rank_z+1)%m_numprocs_z, loc_pcomm_z_tag, m_comm_z, &status);

        int offset_beam = 0;
        for (int ibeam = 0; ibeam < nbeams; ibeam++){
            auto& ptile = m_multi_beam.getBeam(ibeam);
            const int np = np_rcv[ibeam];
            auto old_size = ptile.numParticles();
            auto new_size = old_size + np;
            ptile.resize(new_size);
            const auto ptd = ptile.getParticleTileData();

            const amrex::Gpu::DeviceVector<int> comm_real(AMREX_SPACEDIM + m_multi_beam.NumRealComps(), 1);
            const amrex::Gpu::DeviceVector<int> comm_int (AMREX_SPACEDIM + m_multi_beam.NumIntComps(),  1);
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

        amrex::Gpu::streamSynchronize();
        amrex::The_Pinned_Arena()->free(recv_buffer);
    }

    // Receive laser
    {
        if (only_ghost) return;
        if (!m_multi_laser.m_use_laser) return;
        AMREX_ALWAYS_ASSERT(nz_laser > 0);
        amrex::FArrayBox& laser_fab = m_multi_laser.getFAB();
        amrex::Array4<amrex::Real> laser_arr = laser_fab.array();
        const amrex::Box& bx = laser_fab.box(); // does not include ghost cells
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            bx.bigEnd(2)-bx.smallEnd(2)+1 == nz_laser,
            "Laser requires all sub-domains to be the same size, i.e., nz%nrank=0");
        const std::size_t nreals = bx.numPts()*laser_fab.nComp();
        MPI_Status lstatus;

        if (m_multi_laser.is3dOnHost()) {
            // Directly receive envelope in laser fab
            MPI_Recv(laser_fab.dataPtr(), nreals,
                     amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                     (m_rank_z+1)%m_numprocs_z, lcomm_z_tag, m_comm_z, &lstatus);
        } else {
            // Receive envelope in a host buffer, and copy to laser fab on device
            auto lrecv_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
                (sizeof(amrex::Real)*nreals);
            MPI_Recv(lrecv_buffer, nreals,
                     amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                     (m_rank_z+1)%m_numprocs_z, lcomm_z_tag, m_comm_z, &lstatus);

            auto const buf = amrex::makeArray4(lrecv_buffer, bx, laser_fab.nComp());
            amrex::ParallelFor
                (bx, laser_fab.nComp(),
                 [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                 {
                     laser_arr(i,j,k,n) = buf(i,j,k,n);
                 });
            amrex::Gpu::streamSynchronize();
            amrex::The_Pinned_Arena()->free(lrecv_buffer);
        }
    }
#endif
}

void
Hipace::Notify (const int step, const int it,
                const amrex::Vector<BeamBins>& bins, bool only_ghost)
{
    HIPACE_PROFILE("Hipace::Notify()");

#ifdef AMREX_USE_MPI
    constexpr int lev = 0;
    if (m_numprocs_z == 1) return;

    const bool only_time = m_physical_time >= m_max_time;
    NotifyFinish(it, only_ghost, only_time); // finish the previous send
    int nz_laser = 0;
    if (m_multi_laser.m_use_laser){
        const amrex::Box& laser_bx = m_multi_laser.getFAB().box();
        nz_laser = laser_bx.bigEnd(2) - laser_bx.smallEnd(2) + 1;
    }
    const int nbeams = m_multi_beam.get_nbeams();
    const int nint = nbeams + 2;

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
        m_tsend_buffer = m_physical_time + m_dt;
        MPI_Isend(&m_tsend_buffer, 1, amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                  (m_rank_z-1+m_numprocs_z)%m_numprocs_z, tcomm_z_tag, m_comm_z, &m_tsend_request);
    }

    if (only_time) return;

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
    np_snd[nbeams+1] = nz_laser;

    // Each rank sends data downstream, except rank 0 who sends data to m_numprocs_z-1
    const int loc_ncomm_z_tag = only_ghost ? ncomm_z_tag_ghost : ncomm_z_tag;
    MPI_Request* loc_nsend_request = only_ghost ? &m_nsend_request_ghost : &m_nsend_request;
    MPI_Isend(np_snd.dataPtr(), nint, amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
              (m_rank_z-1+m_numprocs_z)%m_numprocs_z, loc_ncomm_z_tag, m_comm_z, loc_nsend_request);

    // Send beam particles. Currently only one tile.
    {
        const amrex::Long np_total = std::accumulate(np_snd.begin(), np_snd.begin()+nbeams, 0);
        if (np_total == 0 && !m_multi_laser.m_use_laser) return;
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

            const amrex::Gpu::DeviceVector<int> comm_real(AMREX_SPACEDIM + m_multi_beam.NumRealComps(), 1);
            const amrex::Gpu::DeviceVector<int> comm_int (AMREX_SPACEDIM + m_multi_beam.NumIntComps(),  1);
            const auto p_comm_real = comm_real.data();
            const auto p_comm_int = comm_int.data();
            const auto p_psend_buffer = psend_buffer + offset_beam*psize;

            BeamBins::index_type const * const indices = bins[ibeam].permutationPtr();
            BeamBins::index_type const * const offsets = bins[ibeam].offsetsPtrCpu();
            BeamBins::index_type cell_start = 0;

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
            amrex::Gpu::streamSynchronize();

            // Delete beam particles that we just sent from the particle array
            if (!only_ghost) ptile.resize(offset_box);
            offset_beam += np;
        } // here

        const int loc_pcomm_z_tag = only_ghost ? pcomm_z_tag_ghost : pcomm_z_tag;
        MPI_Request* loc_psend_request = only_ghost ? &m_psend_request_ghost : &m_psend_request;
        // Each rank sends data downstream, except rank 0 who sends data to m_numprocs_z-1

        // Make datatype the same size as one particle, so MAX_INT particles can be sent
        MPI_Datatype one_particle_size{};
        MPI_Type_contiguous(psize, amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                            &one_particle_size);
        MPI_Type_commit(&one_particle_size);

        MPI_Isend(psend_buffer, np_total, one_particle_size,
                  (m_rank_z-1+m_numprocs_z)%m_numprocs_z, loc_pcomm_z_tag, m_comm_z, loc_psend_request);
    }

    // Send laser data
    {
        if (only_ghost) return;
        if (!m_multi_laser.m_use_laser) return;
        const amrex::FArrayBox& laser_fab = m_multi_laser.getFAB();
        amrex::Array4<amrex::Real const> const& laser_arr = laser_fab.array();
        const amrex::Box& lbx = laser_fab.box(); // does not include ghost cells
        const std::size_t nreals = lbx.numPts()*laser_fab.nComp();
        m_lsend_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real)*nreals);
        if (m_multi_laser.is3dOnHost()) {
            amrex::Gpu::streamSynchronize();
            // Copy from laser envelope 3D array (on host) to MPI buffer (on host)
            laser_fab.copyToMem<amrex::RunOn::Host>(lbx, 0, laser_fab.nComp(), m_lsend_buffer);
        } else {
            // Copy from laser envelope 3D array (on device) to MPI buffer (on host)
            auto const buf = amrex::makeArray4(m_lsend_buffer, lbx, laser_fab.nComp());
            amrex::ParallelFor
                (lbx, laser_fab.nComp(),
                 [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                 {
                     buf(i,j,k,n) = laser_arr(i,j,k,n);
                 });
            amrex::Gpu::streamSynchronize();
        }
        MPI_Isend(m_lsend_buffer, nreals,
                  amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                  (m_rank_z-1+m_numprocs_z)%m_numprocs_z, lcomm_z_tag, m_comm_z, &m_lsend_request);
    }
#endif
}

void
Hipace::NotifyFinish (const int it, bool only_ghost, bool only_time)
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
            AMREX_ALWAYS_ASSERT(m_dt >= 0.);
            if (m_tsend_buffer >= m_initial_time) {
                MPI_Status status;
                MPI_Wait(&m_tsend_request, &status);
                m_tsend_buffer = m_initial_time - 1.;
            }
            if (only_time) return;
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
        if (m_lsend_buffer) {
            MPI_Status status;
            MPI_Wait(&m_lsend_request, &status);
            amrex::The_Pinned_Arena()->free(m_lsend_buffer);
            m_lsend_buffer = nullptr;
        }
    }
#endif
}

void
Hipace::ResizeFDiagFAB (const int it)
{
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        amrex::Box local_box = boxArray(lev)[it];
        amrex::Box domain = boxArray(lev).minimalBox();

        if (lev == 1) {
            // boxArray(1) is not correct in z direction. We need to manually enforce a
            // parent/child relationship between lev_0 and lev_1 boxes in z
            const amrex::Box& bx_lev0 = boxArray(0)[it];
            const int ref_ratio_z = GetRefRatio(lev)[Direction::z];

            // This seems to be required for some reason
            domain.setBig(Direction::z, domain.bigEnd(Direction::z) - ref_ratio_z);

            // Ensuring the IO boxes on level 1 are aligned with the boxes on level 0
            local_box.setSmall(Direction::z, amrex::max(domain.smallEnd(Direction::z),
                               ref_ratio_z*bx_lev0.smallEnd(Direction::z)));
            local_box.setBig  (Direction::z, amrex::min(domain.bigEnd(Direction::z),
                               ref_ratio_z*bx_lev0.bigEnd(Direction::z)+(ref_ratio_z-1)));
        }

        m_diags.ResizeFDiagFAB(local_box, domain, lev, Geom(lev));
    }
}

amrex::IntVect
Hipace::GetRefRatio (int lev)
{
    if (lev==0) {
        return amrex::IntVect{1,1,1};
    } else {
        return GetInstance().ref_ratio[lev-1];
    }
}

void
Hipace::FillDiagnostics (const int lev, int i_slice)
{
    if (m_diags.hasField()[lev]) {
        m_fields.Copy(lev, i_slice, m_diags.getGeom()[lev], m_diags.getF(lev),
                      m_diags.getF(lev).box(), Geom(lev),
                      m_diags.getCompsIdx(), m_diags.getNFields(), m_multi_laser);
    }
}

void
Hipace::WriteDiagnostics (int output_step, const int it, const OpenPMDWriterCallType call_type)
{
    HIPACE_PROFILE("Hipace::WriteDiagnostics()");

    // Dump every m_output_period steps and after last step
    if (m_output_period < 0 ||
        (!(m_physical_time == m_max_time) && !(output_step == m_max_step)
         && output_step % m_output_period != 0 ) ) return;

    // assumption: same order as in struct enum Field Comps
    amrex::Vector< std::string > varnames = getDiagComps();
    if (m_diags.doLaser()) varnames.push_back("laser_real");
    if (m_diags.doLaser()) varnames.push_back("laser_imag");
    const amrex::Vector< std::string > beamnames = getDiagBeamNames();

#ifdef HIPACE_USE_OPENPMD
    amrex::Gpu::streamSynchronize();
    m_openpmd_writer.WriteDiagnostics(getDiagF(), m_multi_beam, getDiagGeom(), m_diags.hasField(),
                        m_physical_time, output_step, finestLevel()+1, getDiagSliceDir(), varnames,
                        beamnames, it, m_box_sorters, geom, call_type);
#else
    amrex::ignore_unused(it, call_type);
    amrex::Print()<<"WARNING: HiPACE++ compiled without openPMD support, the simulation has no I/O.\n";
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
