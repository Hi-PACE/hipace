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
#include "utils/DeprecatedInput.H"
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

int Hipace_early_init::m_depos_order_xy = 2;
int Hipace_early_init::m_depos_order_z = 0;
int Hipace_early_init::m_depos_derivative_type = 2;
bool Hipace_early_init::m_outer_depos_loop = false;

Hipace* Hipace::m_instance = nullptr;

bool Hipace::m_normalized_units = false;
int Hipace::m_max_step = 0;
amrex::Real Hipace::m_dt = 0.0;
amrex::Real Hipace::m_max_time = std::numeric_limits<amrex::Real>::infinity();
amrex::Real Hipace::m_physical_time = 0.0;
amrex::Real Hipace::m_initial_time = 0.0;
int Hipace::m_verbose = 0;
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

    queryWithParser(pph, "depos_order_xy", m_depos_order_xy);
    queryWithParser(pph, "depos_order_z", m_depos_order_z);
    queryWithParser(pph, "depos_derivative_type", m_depos_derivative_type);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_depos_order_xy != 0 || m_depos_derivative_type != 0,
                            "Analytic derivative with depos_order=0 would vanish");
    queryWithParser(pph, "outer_depos_loop", m_outer_depos_loop);

    amrex::ParmParse pp_amr("amr");
    int max_level = 0;
    queryWithParser(pp_amr, "max_level", max_level);
    m_N_level = max_level + 1;
}

Hipace&
Hipace::GetInstance ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_instance, "instance has not been initialized yet");
    return *m_instance;
}

Hipace::Hipace () :
    Hipace_early_init(this),
    m_fields(m_N_level),
    m_multi_beam(),
    m_multi_plasma(),
    m_adaptive_time_step(m_multi_beam.get_nbeams()),
    m_multi_laser(),
    m_diags(m_N_level)
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
    queryWithParser(pph, "predcorr_B_error_tolerance", m_predcorr_B_error_tolerance);
    queryWithParser(pph, "predcorr_max_iterations", m_predcorr_max_iterations);
    queryWithParser(pph, "predcorr_B_mixing_factor", m_predcorr_B_mixing_factor);
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

    std::string solver = "explicit";
    queryWithParser(pph, "bxby_solver", solver);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solver == "predictor-corrector" ||
        solver == "explicit",
        "hipace.bxby_solver must be explicit or predictor-corrector");
    m_explicit = solver == "explicit" ? true : false;
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

#ifdef AMREX_USE_MPI
    queryWithParser(pph, "skip_empty_comms", m_skip_empty_comms);
    int myproc = amrex::ParallelDescriptor::MyProc();
    m_rank_z = myproc/(m_numprocs_x*m_numprocs_y);
    MPI_Comm_split(amrex::ParallelDescriptor::Communicator(), m_rank_z, myproc, &m_comm_xy);
    MPI_Comm_rank(m_comm_xy, &m_rank_xy);
    MPI_Comm_split(amrex::ParallelDescriptor::Communicator(), m_rank_xy, myproc, &m_comm_z);
#endif

    MakeGeometry();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_N_level == 1 || !m_multi_beam.AnySpeciesSalame(),
        "Cannot use SALAME algorithm with mesh refinement");

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
#ifdef HIPACE_USE_AB5_PUSH
    amrex::Print() << "using the Adams-Bashforth plasma particle pusher\n";
#else
    amrex::Print() << "using the leapfrog plasma particle pusher\n";
#endif

    for (int lev=0; lev<m_N_level; ++lev) {
        m_fields.AllocData(lev, m_3D_geom[lev], m_slice_ba[lev], m_slice_dm[lev],
                       m_multi_plasma.m_sort_bin_size);
        if (lev==0) {
            m_multi_laser.InitData(m_slice_ba[0], m_slice_dm[0]); // laser inits only on level 0
        }
        m_diags.Initialize(lev, m_multi_laser.m_use_laser);
    }

    m_initial_time = m_multi_beam.InitData(m_3D_geom[0]);
    if (Hipace::HeadRank()) {
        m_adaptive_time_step.Calculate(m_physical_time, m_dt, m_multi_beam,
                                       m_multi_plasma, m_3D_geom[0], m_fields);
        m_adaptive_time_step.CalculateFromDensity(m_physical_time, m_dt, m_multi_plasma);
    }
#ifdef AMREX_USE_MPI
    m_adaptive_time_step.BroadcastTimeStep(m_dt, m_comm_z, m_numprocs_z);
#endif
    m_physical_time = m_initial_time;
#ifdef AMREX_USE_MPI
    m_physical_time += m_dt * (m_numprocs_z-1-amrex::ParallelDescriptor::MyProc());
#endif
    if (!Hipace::HeadRank()) {
        m_adaptive_time_step.Calculate(m_physical_time, m_dt, m_multi_beam,
                                       m_multi_plasma, m_3D_geom[0], m_fields);
        m_adaptive_time_step.CalculateFromDensity(m_physical_time, m_dt, m_multi_plasma);
    }
    m_fields.checkInit();
}

void
Hipace::MakeGeometry ()
{
    m_3D_geom.resize(m_N_level);
    m_3D_dm.resize(m_N_level);
    m_3D_ba.resize(m_N_level);
    m_slice_geom.resize(m_N_level);
    m_slice_dm.resize(m_N_level);
    m_slice_ba.resize(m_N_level);

    // make 3D Geometry, BoxArray, DistributionMapping on level 0
    amrex::ParmParse pp_amr("amr");
    std::array<int, 3> n_cells {0, 0, 0};
    getWithParser(pp_amr, "n_cell", n_cells);
    const amrex::Box domain_3D{amrex::IntVect(0,0,0), n_cells.data()};

    // this will get prob_lo, prob_hi and is_periodic from the input file
    m_3D_geom[0].define(domain_3D, nullptr, amrex::CoordSys::cartesian, nullptr);

    const int n_boxes = (m_boxes_in_z == 1) ? m_numprocs_z : m_boxes_in_z;

    amrex::BoxList bl{};
    amrex::Vector<int> procmap{};
    for (int i=0; i<n_boxes; ++i) {
        bl.push_back(
            amrex::Box(domain_3D)
                .setSmall(2, domain_3D.smallEnd(2) + (i*domain_3D.length(2))/n_boxes )
                .setBig(2, domain_3D.smallEnd(2) + ((i+1)*domain_3D.length(2))/n_boxes -1 )
        );
        procmap.push_back(
            (i*m_numprocs_z)/n_boxes
        );
    }
    m_3D_ba[0].define(bl);
    m_3D_dm[0].define(procmap);

    // make 3D Geometry, BoxArray, DistributionMapping on level >= 1
    for (int lev=1; lev<m_N_level; ++lev) {
        amrex::ParmParse pp_mrlev("mr_lev" + std::to_string(lev));

        // get n_cell in x and y direction, z direction is calculated from the patch size
        std::array<int, 2> n_cells_lev {0, 0};
        std::array<amrex::Real, 3> patch_lo_lev {0, 0, 0};
        std::array<amrex::Real, 3> patch_hi_lev {0, 0, 0};
        getWithParser(pp_mrlev, "n_cell", n_cells_lev);
        getWithParser(pp_mrlev, "patch_lo", patch_lo_lev);
        getWithParser(pp_mrlev, "patch_hi", patch_hi_lev);

        const amrex::Real pos_offset_z = GetPosOffset(2, m_3D_geom[0], m_3D_geom[0].Domain());

        const int zeta_lo = std::max( m_3D_geom[0].Domain().smallEnd(2),
            int(amrex::Math::round((patch_lo_lev[2] - pos_offset_z) * m_3D_geom[0].InvCellSize(2)))
        );

        const int zeta_hi = std::min( m_3D_geom[0].Domain().bigEnd(2),
            int(amrex::Math::round((patch_hi_lev[2] - pos_offset_z) * m_3D_geom[0].InvCellSize(2)))
        );

        patch_lo_lev[2] = (zeta_lo-0.5)*m_3D_geom[0].CellSize(2) + pos_offset_z;
        patch_hi_lev[2] = (zeta_hi+0.5)*m_3D_geom[0].CellSize(2) + pos_offset_z;

        const amrex::Box domain_3D_lev{amrex::IntVect(0,0,zeta_lo),
            amrex::IntVect(n_cells_lev[0]-1, n_cells_lev[1]-1, zeta_hi)};

        // non-periodic because it is internal
        m_3D_geom[lev].define(domain_3D_lev, amrex::RealBox(patch_lo_lev, patch_hi_lev),
                              amrex::CoordSys::cartesian, {0, 0, 0});

        amrex::BoxList bl_lev{};
        amrex::Vector<int> procmap_lev{};
        for (int i=0; i<n_boxes; ++i) {
            if (m_3D_ba[0][i].smallEnd(2) > zeta_hi || m_3D_ba[0][i].bigEnd(2) < zeta_lo) {
                continue;
            }
            // enforce parent-child relationship with level 0 BoxArray
            bl_lev.push_back(
                amrex::Box(domain_3D_lev)
                    .setSmall(2, std::max(domain_3D_lev.smallEnd(2), m_3D_ba[0][i].smallEnd(2)) )
                    .setBig(2, std::min(domain_3D_lev.bigEnd(2), m_3D_ba[0][i].bigEnd(2)) )
            );
            procmap_lev.push_back(
                (i*m_numprocs_z)/n_boxes
            );
        }
        m_3D_ba[lev].define(bl_lev);
        m_3D_dm[lev].define(procmap_lev);
    }

    // make slice Geometry, BoxArray, DistributionMapping every level
    for (int lev=0; lev<m_N_level; ++lev) {
        amrex::Box slice_box = m_3D_geom[lev].Domain();
        slice_box.setSmall(2, 0);
        slice_box.setBig(2, 0);
        amrex::RealBox slice_realbox = m_3D_geom[lev].ProbDomain();
        slice_realbox.setLo(2, 0.);
        slice_realbox.setHi(2, m_3D_geom[lev].CellSize(2));

        m_slice_geom[lev].define(slice_box, slice_realbox, amrex::CoordSys::cartesian,
                                 m_3D_geom[lev].isPeriodic());
        m_slice_ba[lev].define(slice_box);
        m_slice_dm[lev].define(amrex::Vector<int>({amrex::ParallelDescriptor::MyProc()}));
    }
}

void
Hipace::Evolve ()
{
    HIPACE_PROFILE("Hipace::Evolve()");
    const int rank = amrex::ParallelDescriptor::MyProc();
    m_multi_beam.sortParticlesByBox(m_3D_ba[0], m_3D_geom[0]);

    // now each rank starts with its own time step and writes to its own file. Highest rank starts with step 0
    for (int step = m_numprocs_z - 1 - m_rank_z; step <= m_max_step; step += m_numprocs_z)
    {
        ResetAllQuantities();

        // Loop over longitudinal boxes on this rank, from head to tail
        const int n_boxes = (m_boxes_in_z == 1) ? m_numprocs_z : m_boxes_in_z;
        for (int it = n_boxes-1; it >= 0; --it)
        {
            m_multi_beam.SetIbox(it);

            const amrex::Box& bx = m_3D_ba[0][it];

            if (m_multi_laser.m_use_laser) {
                AMREX_ALWAYS_ASSERT(!m_adaptive_time_step.m_do_adaptive_time_step);
                AMREX_ALWAYS_ASSERT(m_multi_plasma.GetNPlasmas() <= 1);
                // Before that, the 3D fields of the envelope are not initialized (not even allocated).
                m_multi_laser.Init3DEnvelope(step, bx, m_3D_geom[0]);
            }

            Wait(step, it);
            if (it == n_boxes-1) {
                // Only reset plasma after receiving time step, to use proper density
                m_multi_plasma.InitData(m_slice_ba, m_slice_dm, m_slice_geom, m_3D_geom);

                // Even if level 1 doesn't start on the first slice,
                // we need to deposit a neutralizing background now
                // Use to slice -1 to tag to the finest level of any slice
                // to deposit the neutralizing background
                m_multi_plasma.TagByLevel(m_N_level, m_3D_geom, -1);

                /* Store charge density of (immobile) ions into WhichSlice::RhoIons */
                for (int lev=0; lev<m_N_level; ++lev) {
                    if (m_do_tiling) {
                        m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);
                    }
                    m_multi_plasma.DepositNeutralizingBackground(
                        m_fields, m_multi_laser, WhichSlice::RhoIons, m_3D_geom, lev);
                }
            }

            if (m_physical_time >= m_max_time) {
                Notify(step, it); // just send signal to finish simulation
                if (m_physical_time > m_max_time) break;
            }
            m_adaptive_time_step.CalculateFromDensity(m_physical_time, m_dt, m_multi_plasma);

            // adjust time step to reach max_time
            m_dt = std::min(m_dt, m_max_time - m_physical_time);

#ifdef HIPACE_USE_OPENPMD
            // need correct physical time for this check
            if (it == n_boxes-1
                && m_diags.hasAnyOutput(step, m_max_step, m_physical_time, m_max_time)) {
                m_openpmd_writer.InitDiagnostics();
            }
#endif

            if (m_verbose>=1 && it==n_boxes-1) std::cout<<"Rank "<<rank<<" started  step "<<step
                                    <<" at time = "<<m_physical_time<< " with dt = "<<m_dt<<'\n';


            m_multi_beam.sortParticlesByBox(m_3D_ba[0], m_3D_geom[0]);
            m_leftmost_box_snd = std::min(leftmostBoxWithParticles(), m_leftmost_box_snd);

            WriteDiagnostics(step, it, OpenPMDWriterCallType::beams);

            m_multi_beam.StoreNRealParticles();
            // Copy particles in box it-1 in the ghost buffer.
            // This handles both beam initialization and particle slippage.
            if (it>0) m_multi_beam.PackLocalGhostParticles(it-1);

            ResizeFDiagFAB(it, step);

            m_multi_beam.findParticlesInEachSlice(it, bx, m_3D_geom[0]);
            AMREX_ALWAYS_ASSERT( bx.bigEnd(Direction::z) >= bx.smallEnd(Direction::z) + 2 );
            // Solve head slice
            SolveOneSlice(bx.bigEnd(Direction::z), bx.length(Direction::z) - 1, step);
            // Notify ghost slice
            if (it<m_numprocs_z-1) Notify(step, it, true);
            // Solve central slices
            for (int isl = bx.bigEnd(Direction::z)-1; isl > bx.smallEnd(Direction::z); --isl){
                SolveOneSlice(isl, isl - bx.smallEnd(Direction::z), step);
            };
            // Receive ghost slice
            if (it>0) Wait(step, it, true);
            CheckGhostSlice(it);
            // Solve tail slice. Consume ghost particles.
            SolveOneSlice(bx.smallEnd(Direction::z), 0, step);
            // Delete ghost particles
            m_multi_beam.RemoveGhosts();

            if (m_physical_time < m_max_time) {
                m_adaptive_time_step.Calculate(
                    m_physical_time, m_dt, m_multi_beam, m_multi_plasma,
                    m_3D_geom[0], m_fields, it, false);
            } else {
                m_dt = 2.*m_max_time;
            }


            // averaging predictor corrector loop diagnostics
            m_predcorr_avg_iterations /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));
            m_predcorr_avg_B_error /= (bx.bigEnd(Direction::z) + 1 - bx.smallEnd(Direction::z));

            WriteDiagnostics(step, it, OpenPMDWriterCallType::fields);
            Notify(step, it);
        }

        m_multi_beam.InSituWriteToFile(step, m_physical_time, m_3D_geom[0]);

        // printing and resetting predictor corrector loop diagnostics
        if (m_verbose>=2 && !m_explicit) amrex::AllPrint() << "Rank " << rank <<
                                ": avg. number of iterations " << m_predcorr_avg_iterations <<
                                " avg. transverse B field error " << m_predcorr_avg_B_error << "\n";
        m_predcorr_avg_iterations = 0.;
        m_predcorr_avg_B_error = 0.;

        m_physical_time += m_dt;

#ifdef HIPACE_USE_OPENPMD
        m_openpmd_writer.reset();
#endif
    }
}

void
Hipace::SolveOneSlice (int islice, const int islice_local, int step)
{
    HIPACE_PROFILE("Hipace::SolveOneSlice()");

    // Between this push and the corresponding pop at the end of this
    // for function, the parallelcontext is the transverse communicator
    amrex::ParallelContext::push(m_comm_xy);

    m_multi_beam.InSituComputeDiags(step, islice, islice_local);

    // Get this laser slice from the 3D array
    m_multi_laser.Copy(islice, false);

    m_multi_beam.TagByLevel(m_N_level, m_3D_geom, WhichSlice::This, islice, islice_local);
    m_multi_beam.TagByLevel(m_N_level, m_3D_geom, WhichSlice::Next, islice, islice_local);
    m_multi_plasma.TagByLevel(m_N_level, m_3D_geom, islice);

    for (int lev=0; lev<m_N_level; ++lev) {

        if (lev != 0) {
            // skip all slices which are not existing on level 1
            if (islice < m_3D_geom[lev].Domain().smallEnd(Direction::z) ||
                islice > m_3D_geom[lev].Domain().bigEnd(Direction::z)) {
                continue;
            } else if (islice == m_3D_geom[lev].Domain().bigEnd(Direction::z)) {
                // first slice of level 1 (islice goes backwards)
                // iterpolate jx_beam and jy_beam from level 0 to level 1
                m_fields.LevelUp(m_3D_geom, lev, WhichSlice::Previous1, "jx_beam");
                m_fields.LevelUp(m_3D_geom, lev, WhichSlice::Previous1, "jy_beam");
                m_fields.LevelUp(m_3D_geom, lev, WhichSlice::This, "jx_beam");
                m_fields.LevelUp(m_3D_geom, lev, WhichSlice::This, "jy_beam");
                m_fields.duplicate(lev, WhichSlice::This, {"jx"     , "jy"     },
                                        WhichSlice::This, {"jx_beam", "jy_beam"});
            }
        }

        // reorder plasma before TileSort
        m_multi_plasma.ReorderParticles(islice);

        if (m_do_tiling) m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);

        if (m_explicit) {
            ExplicitSolveOneSubSlice(lev, step, islice, islice_local);
        } else {
            PredictorCorrectorSolveOneSubSlice(lev, step, islice, islice_local);
        }

        FillDiagnostics(lev, islice);

        m_multi_plasma.DoFieldIonization(lev, m_3D_geom[lev], m_fields);

        if (m_multi_plasma.IonizationOn() && m_do_tiling) {
            m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);
        }

        // Push plasma particles
        m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, m_3D_geom, false, lev);

        // Push beam particles
        m_multi_beam.AdvanceBeamParticlesSlice(m_fields, m_3D_geom[lev], lev, islice_local);

    } // end for (int lev=0; lev<m_N_level; ++lev)

    // collisions for all particles calculated on level 0
    m_multi_plasma.doCoulombCollision(0, m_slice_geom[0].Domain(), m_slice_geom[0]);

    // Advance laser slice by 1 step and store result to 3D array
    // no MR for laser
    m_multi_laser.AdvanceSlice(m_fields, m_3D_geom[0], m_dt, step);
    m_multi_laser.Copy(islice, true);

    // shift all levels
    for (int lev=0; lev<m_N_level; ++lev) {
        if (lev != 0) {
            // skip all slices which are not existing on level 1
            if (islice < m_3D_geom[lev].Domain().smallEnd(Direction::z) ||
                islice > m_3D_geom[lev].Domain().bigEnd(Direction::z)) {
                continue;
            }
        }

        m_fields.ShiftSlices(lev);
    }

    // After this, the parallel context is the full 3D communicator again
    amrex::ParallelContext::pop();
}


void
Hipace::ExplicitSolveOneSubSlice (const int lev, const int step,
                                  const int islice, const int islice_local)
{
    // Set all quantities to 0 except:
    // Bx and By: the previous slice serves as initial guess.
    // jx_beam and jy_beam are used from the previous "Next" slice
    // jx and jy are initially set to jx_beam and jy_beam
    m_fields.setVal(0., lev, WhichSlice::This, "chi", "Sy", "Sx", "ExmBy", "EypBx", "Ez",
        "Bz", "Psi", "jz_beam", "rho_beam", "jz", "rho");

    // deposit jx, jy, jz, rho and chi for all plasmas
    m_multi_plasma.DepositCurrent(
        m_fields, m_multi_laser, WhichSlice::This, true, true, true, true, m_3D_geom, lev);

    m_fields.setVal(0., lev, WhichSlice::Next, "jx_beam", "jy_beam");
    // deposit jx_beam and jy_beam in the Next slice
    m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step, islice_local,
        m_do_beam_jx_jy_deposition, false, false, WhichSlice::Next);
    // need to exchange jx_beam jy_beam
    m_fields.FillBoundary(m_3D_geom[lev].periodicity(),lev, WhichSlice::Next, "jx_beam", "jy_beam");

    m_fields.AddRhoIons(lev);

    // deposit jz_beam and maybe rho_beam on This slice
    m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step, islice_local,
        false, true, m_do_beam_jz_minus_rho, WhichSlice::This);

    FillBoundaryChargeCurrents(lev);

    // interpolate jx and jy to level 1 in the domain edges and
    // also inside ghost cells to account for x and y derivative
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "jx",
        m_fields.m_slices_nguards, -m_fields.m_slices_nguards);
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "jy",
        m_fields.m_slices_nguards, -m_fields.m_slices_nguards);

    m_fields.SolvePoissonExmByAndEypBx(m_3D_geom, lev);
    m_fields.SolvePoissonEz(m_3D_geom, lev);
    m_fields.SolvePoissonBz(m_3D_geom, lev);

    // deposit grid current into jz_beam
    m_grid_current.DepositCurrentSlice(m_fields, m_3D_geom[lev], lev, islice);
    // No FillBoundary because grid current only deposits in the middle of the field

    // Set Sx and Sy to beam contribution
    InitializeSxSyWithBeam(lev);

    // Deposit Sx and Sy for every plasma species
    m_multi_plasma.ExplicitDeposition(m_fields, m_multi_laser, m_3D_geom, lev);

    // Solves Bx, By using Sx, Sy and chi
    ExplicitMGSolveBxBy(lev, WhichSlice::This);

    if (m_multi_beam.isSalameNow(step, islice_local)) {
        // Modify the beam particle weights on this slice to flatten Ez.
        // As the beam current is modified, Bx and By are also recomputed.
        SalameModule(this, m_salame_n_iter, m_salame_do_advance, m_salame_last_slice,
                     m_salame_overloaded, lev, step, islice, islice_local);
    }

    // Push beam and plasma in SolveOneSlice
}

void
Hipace::PredictorCorrectorSolveOneSubSlice (const int lev, const int step,
                                            const int islice, const int islice_local)
{
    m_fields.setVal(0., lev, WhichSlice::This,
        "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi");
    if (m_use_laser) {
        m_fields.setVal(0., lev, WhichSlice::This, "chi");
    }

    // deposit jx jy jz rho and maybe chi
    m_multi_plasma.DepositCurrent(m_fields, m_multi_laser, WhichSlice::This,
        true, true, true, m_use_laser, m_3D_geom, lev);

    m_fields.AddRhoIons(lev);

    FillBoundaryChargeCurrents(lev);

    if (!m_do_beam_jz_minus_rho) {
        m_fields.SolvePoissonExmByAndEypBx(m_3D_geom, lev);
    }

    // deposit jx jy jz and maybe rho on This slice
    m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step, islice_local,
                                     m_do_beam_jx_jy_deposition, true,
                                     m_do_beam_jz_minus_rho, WhichSlice::This);

    if (m_do_beam_jz_minus_rho) {
        m_fields.SolvePoissonExmByAndEypBx(m_3D_geom, lev);
    }

    // deposit grid current into jz_beam
    m_grid_current.DepositCurrentSlice(m_fields, m_3D_geom[lev], lev, islice);

    FillBoundaryChargeCurrents(lev);

    m_fields.SolvePoissonEz(m_3D_geom, lev);
    m_fields.SolvePoissonBz(m_3D_geom, lev);

    // Solves Bx and By in the current slice and modifies the force terms of the plasma particles
    PredictorCorrectorLoopToSolveBxBy(islice, islice_local, lev, step);

    // Push beam and plasma in SolveOneSlice
}

void
Hipace::ResetAllQuantities ()
{
    HIPACE_PROFILE("Hipace::ResetAllQuantities()");

    if (m_use_laser) ResetLaser();

    for (int lev=0; lev<m_N_level; ++lev) {
        if (m_fields.getSlices(lev).nComp() != 0) {
            m_fields.getSlices(lev).setVal(0., m_fields.m_slices_nguards);
        }
    }
}

void
Hipace::ResetLaser ()
{
    HIPACE_PROFILE("Hipace::ResetLaser()");

    m_multi_laser.getSlices().setVal(0.);
}

void
Hipace::FillBoundaryChargeCurrents (int lev) {
    if (!m_fields.m_extended_solve) {
        if (m_explicit) {
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::This,
                "jx_beam", "jy_beam", "jz_beam", "rho_beam", "jx", "jy", "jz", "rho");
        } else {
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::This,
                "jx", "jy", "jz", "rho");
        }
    }
}

void
Hipace::InitializeSxSyWithBeam (const int lev)
{
    HIPACE_PROFILE("Hipace::InitializeSxSyWithBeam()");
    using namespace amrex::literals;

    amrex::MultiFab& slicemf = m_fields.getSlices(lev);

    const amrex::Real dx = m_3D_geom[lev].CellSize(Direction::x);
    const amrex::Real dy = m_3D_geom[lev].CellSize(Direction::y);
    const amrex::Real dz = m_3D_geom[lev].CellSize(Direction::z);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        amrex::Box const& bx = mfi.tilebox();

        Array3<amrex::Real> const arr = slicemf.array(mfi);

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
                const amrex::Real dx_jzb = (arr(i+1,j,jzb)-arr(i-1,j,jzb))/(2._rt*dx);
                const amrex::Real dy_jzb = (arr(i,j+1,jzb)-arr(i,j-1,jzb))/(2._rt*dy);
                const amrex::Real dz_jxb = (arr(i,j,prev_jxb)-arr(i,j,next_jxb))/(2._rt*dz);
                const amrex::Real dz_jyb = (arr(i,j,prev_jyb)-arr(i,j,next_jyb))/(2._rt*dz);

                // calculate contribution to Sx and Sy by all beams (same as with PC solver)
                // sy, to compute Bx
                arr(i,j,Sy) =   mu0 * ( - dy_jzb + dz_jyb);
                // sx, to compute By
                arr(i,j,Sx) = - mu0 * ( - dx_jzb + dz_jxb);
            });
    }
}


void
Hipace::ExplicitMGSolveBxBy (const int lev, const int which_slice)
{
    HIPACE_PROFILE("Hipace::ExplicitMGSolveBxBy()");

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

    amrex::MultiFab& slicemf = m_fields.getSlices(lev);
    amrex::MultiFab BxBy (slicemf, amrex::make_alias, Comps[which_slice]["Bx"], 2);
    amrex::MultiFab SySx (slicemf, amrex::make_alias, Comps[which_slice]["Sy"], 2);
    amrex::MultiFab Mult (slicemf, amrex::make_alias, Comps[which_slice_chi]["chi"], ncomp_chi);

    // interpolate Sx, Sy and chi to level 1 in the domain edges.
    // This also accounts for jx_beam, jy_beam
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "Sy",
        m_fields.m_poisson_nguards, -m_fields.m_slices_nguards);
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "Sx",
        m_fields.m_poisson_nguards, -m_fields.m_slices_nguards);
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "chi",
        m_fields.m_poisson_nguards, -m_fields.m_slices_nguards);

    if (lev!=0 && (slicemf.box(0).length(0) % 2 == 0)) {
        // cell centered MG solve: no ghost cells, put boundary condition into source term
        // node centered MG solve: one ghost cell, use boundary condition from there
        m_fields.SetBoundaryCondition(m_3D_geom, lev, "Bx",
                                      m_fields.getField(lev, which_slice, "Sy"));
        m_fields.SetBoundaryCondition(m_3D_geom, lev, "By",
                                      m_fields.getField(lev, which_slice, "Sx"));
    }

    // interpolate Bx and By to level 1 in the ghost cells
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "Bx",
        m_fields.m_slices_nguards, m_fields.m_poisson_nguards);
    m_fields.InterpolateFromLev0toLev1(m_3D_geom, lev, "By",
        m_fields.m_slices_nguards, m_fields.m_poisson_nguards);

#ifdef AMREX_USE_LINEAR_SOLVERS
    if (m_use_amrex_mlmg) {
        // Copy chi to chi2
        m_fields.duplicate(lev, which_slice_chi, {"chi2"}, which_slice_chi, {"chi"});
        amrex::Gpu::streamSynchronize();
        if (m_mlalaplacian.size()<m_N_level) {
            m_mlalaplacian.resize(m_N_level);
            m_mlmg.resize(m_N_level);
        }

        // construct slice geometry
        const amrex::RealBox slice_box{slicemf.boxArray()[0], m_slice_geom[lev].CellSize(),
                                       m_slice_geom[lev].ProbLo()};
        amrex::Geometry slice_geom{slicemf.boxArray()[0], slice_box,
                                   m_slice_geom[lev].CoordInt(), {0,0,0}};

        if (!m_mlalaplacian[lev]){
            // If first call, initialize the MG solver
            amrex::LPInfo lpinfo{};
            lpinfo.setHiddenDirection(2).setAgglomeration(false).setConsolidation(false);

            // make_unique requires explicit types
            m_mlalaplacian[lev] = std::make_unique<amrex::MLALaplacian>(
                amrex::Vector<amrex::Geometry>{slice_geom},
                amrex::Vector<amrex::BoxArray>{slicemf.boxArray()},
                amrex::Vector<amrex::DistributionMapping>{slicemf.DistributionMap()},
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
        AMREX_ALWAYS_ASSERT(slicemf.boxArray().size() == 1);
        if (m_hpmg.size()<m_N_level) {
            m_hpmg.resize(m_N_level);
        }
        if (!m_hpmg[lev]) {
            m_hpmg[lev] = std::make_unique<hpmg::MultiGrid>(m_slice_geom[lev].CellSize(0),
                                                            m_slice_geom[lev].CellSize(1),
                                                            slicemf.boxArray()[0]);
        }
        const int max_iters = 200;
        m_hpmg[lev]->solve1(BxBy[0], SySx[0], Mult[0], m_MG_tolerance_rel, m_MG_tolerance_abs,
                            max_iters, m_MG_verbose);
    }
}

void
Hipace::PredictorCorrectorLoopToSolveBxBy (const int islice, const int islice_local,
                                           const int lev, const int step)
{
    HIPACE_PROFILE("Hipace::PredictorCorrectorLoopToSolveBxBy()");

    amrex::Real relative_Bfield_error_prev_iter = 1.0;
    amrex::Real relative_Bfield_error = m_fields.ComputeRelBFieldError(
        m_fields.getSlices(lev), m_fields.getSlices(lev),
        m_fields.getSlices(lev), m_fields.getSlices(lev),
        Comps[WhichSlice::Previous1]["Bx"], Comps[WhichSlice::Previous1]["By"],
        Comps[WhichSlice::Previous2]["Bx"], Comps[WhichSlice::Previous2]["By"],
        m_3D_geom[lev]);

    /* Guess Bx and By */
    m_fields.InitialBfieldGuess(relative_Bfield_error, m_predcorr_B_error_tolerance, lev);

    if (!m_fields.m_extended_solve) {
        // exchange ExmBy EypBx Ez Bx By Bz
        m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::This,
            "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi");
    }

    /* creating temporary Bx and By arrays for the current and previous iteration */
    amrex::MultiFab Bx_iter(m_fields.getSlices(lev).boxArray(),
                            m_fields.getSlices(lev).DistributionMap(), 1,
                            m_fields.getSlices(lev).nGrowVect());
    amrex::MultiFab By_iter(m_fields.getSlices(lev).boxArray(),
                            m_fields.getSlices(lev).DistributionMap(), 1,
                            m_fields.getSlices(lev).nGrowVect());
    Bx_iter.setVal(0.0, m_fields.m_slices_nguards);
    By_iter.setVal(0.0, m_fields.m_slices_nguards);
    amrex::MultiFab Bx_prev_iter(m_fields.getSlices(lev).boxArray(),
                                 m_fields.getSlices(lev).DistributionMap(), 1,
                                 m_fields.getSlices(lev).nGrowVect());
    amrex::MultiFab::Copy(Bx_prev_iter, m_fields.getSlices(lev),
                          Comps[WhichSlice::This]["Bx"], 0, 1, m_fields.m_slices_nguards);
    amrex::MultiFab By_prev_iter(m_fields.getSlices(lev).boxArray(),
                                 m_fields.getSlices(lev).DistributionMap(), 1,
                                 m_fields.getSlices(lev).nGrowVect());
    amrex::MultiFab::Copy(By_prev_iter, m_fields.getSlices(lev),
                          Comps[WhichSlice::This]["By"], 0, 1, m_fields.m_slices_nguards);

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
        m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, m_3D_geom, true, lev);

        if (m_do_tiling) m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);
        // plasmas deposit jx jy to next temp slice
        m_multi_plasma.DepositCurrent(m_fields, m_multi_laser, WhichSlice::Next,
            true, false, false, false, m_3D_geom, lev);

        // beams deposit jx jy to the next slice
        m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step, islice_local,
            m_do_beam_jx_jy_deposition, false, false, WhichSlice::Next);

        if (!m_fields.m_extended_solve) {
            // need to exchange jx jy jx_beam jy_beam
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::Next,
                "jx", "jy");
        }

        /* Calculate Bx and By */
        m_fields.SolvePoissonBx(Bx_iter, m_3D_geom, lev);
        m_fields.SolvePoissonBy(By_iter, m_3D_geom, lev);

        relative_Bfield_error = m_fields.ComputeRelBFieldError(
            m_fields.getSlices(lev), m_fields.getSlices(lev),
            Bx_iter, By_iter,
            Comps[WhichSlice::This]["Bx"], Comps[WhichSlice::This]["By"],
            0, 0, m_3D_geom[lev]);

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
            // exchange Bx By
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::This,
                "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi");
        }

        // Shift relative_Bfield_error values
        relative_Bfield_error_prev_iter = relative_Bfield_error;
    } /* end of predictor corrector loop */

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
Hipace::Notify (const int step, const int it, bool only_ghost)
{
    HIPACE_PROFILE("Hipace::Notify()");

#ifdef AMREX_USE_MPI
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
                auto& ptile = m_multi_beam.getBeam(ibeam);
                const int offset_box = ptile.m_box_sorter.boxOffsetsPtr()[it];
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

    const amrex::Box& bx = m_3D_ba[0][it];
    for (int ibeam = 0; ibeam < nbeams; ++ibeam)
    {
        np_snd[ibeam] = only_ghost ?
            m_multi_beam.NGhostParticles(ibeam, bx)
            : m_multi_beam.getBeam(ibeam).m_box_sorter.boxCountsPtr()[it];
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
            auto& ptile = m_multi_beam.getBeam(ibeam);
            const auto ptd = ptile.getConstParticleTileData();

            const int offset_box = ptile.m_box_sorter.boxOffsetsPtr()[it];
            const amrex::Long np = np_snd[ibeam];

            const amrex::Gpu::DeviceVector<int> comm_real(AMREX_SPACEDIM + m_multi_beam.NumRealComps(), 1);
            const amrex::Gpu::DeviceVector<int> comm_int (AMREX_SPACEDIM + m_multi_beam.NumIntComps(),  1);
            const auto p_comm_real = comm_real.data();
            const auto p_comm_int = comm_int.data();
            const auto p_psend_buffer = psend_buffer + offset_beam*psize;

            BeamBins::index_type const * const indices = ptile.m_slice_bins.permutationPtr();
            BeamBins::index_type const * const offsets = ptile.m_slice_bins.offsetsPtrCpu();
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
Hipace::ResizeFDiagFAB (const int it, const int step)
{
    for (int lev=0; lev<m_N_level; ++lev) {
        m_diags.ResizeFDiagFAB(m_3D_ba[lev][it], m_3D_geom[lev].Domain(), lev, m_3D_geom[lev],
                               step, m_max_step, m_physical_time, m_max_time);
    }
}

void
Hipace::FillDiagnostics (const int lev, int i_slice)
{
    for (auto& fd : m_diags.getFieldData()) {
        if (fd.m_level == lev && fd.m_has_field) {
            m_fields.Copy(lev, i_slice, fd.m_geom_io, fd.m_F,
                fd.m_F.box(), m_3D_geom[lev],
                fd.m_comps_output_idx, fd.m_nfields,
                fd.m_do_laser, m_multi_laser);
        }
    }
}

void
Hipace::WriteDiagnostics (int output_step, const int it, const OpenPMDWriterCallType call_type)
{
    HIPACE_PROFILE("Hipace::WriteDiagnostics()");

    if (call_type == OpenPMDWriterCallType::beams) {
        if (!m_diags.hasBeamOutput(output_step, m_max_step, m_physical_time, m_max_time)) {
            return;
        }
    } else if (call_type == OpenPMDWriterCallType::fields) {
        if (!m_diags.hasAnyFieldOutput(output_step, m_max_step, m_physical_time, m_max_time)) {
            return;
        }
    }

#ifdef HIPACE_USE_OPENPMD
    amrex::Gpu::streamSynchronize();
    m_openpmd_writer.WriteDiagnostics(m_diags.getFieldData(), m_multi_beam,
                        m_physical_time, output_step, getDiagBeamNames(),
                        it, m_3D_geom, call_type);
#else
    amrex::ignore_unused(it, call_type);
    amrex::Print()<<"WARNING: HiPACE++ compiled without openPMD support, the simulation has no I/O.\n";
#endif
}

int
Hipace::leftmostBoxWithParticles () const
{
    int boxid = m_numprocs_z;
    for (int ibeam=0; ibeam <m_multi_beam.get_nbeams(); ++ibeam) {
        boxid = std::min(m_multi_beam.getBeam(ibeam).m_box_sorter.leftmostBoxWithParticles(),boxid);
    }
    return boxid;
}

void
Hipace::CheckGhostSlice (int it)
{
    HIPACE_PROFILE("Hipace::CheckGhostSlice()");

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
        const amrex::Box& bx = m_3D_ba[0][it];
        const int ilo = bx.smallEnd(Direction::z);

        // Get domain size in physical space
        const amrex::Real dz = m_3D_geom[0].CellSize(Direction::z);
        const amrex::Real dom_lo = m_3D_geom[0].ProbLo(Direction::z);

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
