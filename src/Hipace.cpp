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
    Parser::addConstantsToParser();
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
    if (queryWithParser(pp, "random_seed", seed)) amrex::ResetRandomSeed(seed, seed);

    amrex::ParmParse pph("hipace");

    std::string str_dt {""};
    queryWithParser(pph, "dt", str_dt);
    if (str_dt != "adaptive") queryWithParser(pph, "dt", m_dt);
    queryWithParser(pph, "max_time", m_max_time);
    queryWithParser(pph, "verbose", m_verbose);
    m_numprocs = amrex::ParallelDescriptor::NProcs();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs <= m_max_step+1,
                                     "Please use more or equal time steps than number of ranks");
    queryWithParser(pph, "predcorr_B_error_tolerance", m_predcorr_B_error_tolerance);
    queryWithParser(pph, "predcorr_max_iterations", m_predcorr_max_iterations);
    queryWithParser(pph, "predcorr_B_mixing_factor", m_predcorr_B_mixing_factor);
    queryWithParser(pph, "beam_injection_cr", m_beam_injection_cr);
    queryWithParser(pph, "do_beam_jx_jy_deposition", m_do_beam_jx_jy_deposition);
    queryWithParser(pph, "do_beam_jz_minus_rho", m_do_beam_jz_minus_rho);
    m_deposit_rho = m_diags.needsRho();
    queryWithParser(pph, "deposit_rho", m_deposit_rho);
    queryWithParser(pph, "do_device_synchronize", DO_DEVICE_SYNCHRONIZE);
    bool do_mfi_sync = false;
    queryWithParser(pph, "do_MFIter_synchronize", do_mfi_sync);
    DfltMfi.SetDeviceSync(do_mfi_sync).UseDefaultStream();
    DfltMfiTlng.SetDeviceSync(do_mfi_sync).UseDefaultStream();
    if (amrex::TilingIfNotGPU()) {
        DfltMfiTlng.EnableTiling();
    }
    DeprecatedInput("hipace", "external_ExmBy_slope", "external_E_slope");
    DeprecatedInput("hipace", "external_Ez_slope", "external_E_slope");
    DeprecatedInput("hipace", "external_Ez_uniform", "external_E_uniform");
    queryWithParser(pph, "external_E_uniform", m_external_E_uniform);
    queryWithParser(pph, "external_B_uniform", m_external_B_uniform);
    queryWithParser(pph, "external_E_slope", m_external_E_slope);
    queryWithParser(pph, "external_B_slope", m_external_B_slope);

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

    queryWithParser(pph, "background_density_SI", m_background_density_SI);
    queryWithParser(pph, "comms_buffer_on_gpu", m_comms_buffer_on_gpu);

    MakeGeometry();

    m_use_laser = m_multi_laser.m_use_laser;
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
        m_adaptive_time_step.Calculate(m_physical_time, m_dt, m_multi_beam, m_multi_plasma);
        m_adaptive_time_step.CalculateFromDensity(m_physical_time, m_dt, m_multi_plasma);
    }
#ifdef AMREX_USE_MPI
    m_adaptive_time_step.BroadcastTimeStep(m_dt, amrex::ParallelDescriptor::Communicator(),
                                                 m_numprocs);
#endif
    m_physical_time = m_initial_time;
#ifdef AMREX_USE_MPI
    m_physical_time += m_dt * (m_numprocs-1-amrex::ParallelDescriptor::MyProc());
#endif
    if (!Hipace::HeadRank()) {
        m_adaptive_time_step.Calculate(m_physical_time, m_dt, m_multi_beam, m_multi_plasma);
        m_adaptive_time_step.CalculateFromDensity(m_physical_time, m_dt, m_multi_plasma);
    }
    m_fields.checkInit();

    m_multi_buffer.initialize(m_3D_geom[0].Domain().length(2),
                              m_multi_beam.get_nbeams(),
                              !m_comms_buffer_on_gpu);
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

    amrex::BoxList bl{domain_3D};
    amrex::Vector<int> procmap{amrex::ParallelDescriptor::MyProc()};
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

        const int zeta_lo = std::max( m_3D_geom[lev-1].Domain().smallEnd(2),
            int(amrex::Math::round((patch_lo_lev[2] - pos_offset_z) * m_3D_geom[0].InvCellSize(2)))
        );

        const int zeta_hi = std::min( m_3D_geom[lev-1].Domain().bigEnd(2),
            int(amrex::Math::round((patch_hi_lev[2] - pos_offset_z) * m_3D_geom[0].InvCellSize(2)))
        );

        patch_lo_lev[2] = (zeta_lo-0.5)*m_3D_geom[0].CellSize(2) + pos_offset_z;
        patch_hi_lev[2] = (zeta_hi+0.5)*m_3D_geom[0].CellSize(2) + pos_offset_z;

        const amrex::Box domain_3D_lev{amrex::IntVect(0,0,zeta_lo),
            amrex::IntVect(n_cells_lev[0]-1, n_cells_lev[1]-1, zeta_hi)};

        // non-periodic because it is internal
        m_3D_geom[lev].define(domain_3D_lev, amrex::RealBox(patch_lo_lev, patch_hi_lev),
                              amrex::CoordSys::cartesian, {0, 0, 0});

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_3D_geom[lev].ProbLo(0)-2*m_3D_geom[lev].CellSize(0)-2*m_3D_geom[lev-1].CellSize(0)
            >  m_3D_geom[lev-1].ProbLo(0) &&
            m_3D_geom[lev].ProbHi(0)+2*m_3D_geom[lev].CellSize(0)+2*m_3D_geom[lev-1].CellSize(0)
            <  m_3D_geom[lev-1].ProbHi(0) &&
            m_3D_geom[lev].ProbLo(1)-2*m_3D_geom[lev].CellSize(1)-2*m_3D_geom[lev-1].CellSize(1)
            >  m_3D_geom[lev-1].ProbLo(1) &&
            m_3D_geom[lev].ProbHi(1)+2*m_3D_geom[lev].CellSize(1)+2*m_3D_geom[lev-1].CellSize(1)
            <  m_3D_geom[lev-1].ProbHi(1),
            "Fine MR level must be fully nested inside the next coarsest level "
            "(with a few cells to spare)"
        );

        amrex::BoxList bl_lev{domain_3D_lev};
        amrex::Vector<int> procmap_lev{amrex::ParallelDescriptor::MyProc()};
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

    // now each rank starts with its own time step and writes to its own file. Highest rank starts with step 0
    for (int step = m_numprocs - 1 - rank; step <= m_max_step; step += m_numprocs)
    {
        ResetAllQuantities();

        const amrex::Box& bx = m_3D_ba[0][0];

        if (m_multi_laser.m_use_laser) {
            AMREX_ALWAYS_ASSERT(!m_adaptive_time_step.m_do_adaptive_time_step);
            AMREX_ALWAYS_ASSERT(m_multi_plasma.GetNPlasmas() <= 1);
            // Before that, the 3D fields of the envelope are not initialized (not even allocated).
            m_multi_laser.Init3DEnvelope(step, bx, m_3D_geom[0]);
        }

        m_physical_time = step == 0 ? m_initial_time : m_multi_buffer.get_time();

        if (m_physical_time > m_max_time) {
            if (step+1 <= m_max_step && !m_has_last_step) {
                m_multi_buffer.put_time(m_physical_time);
            }
            break;
        }

        m_adaptive_time_step.CalculateFromDensity(m_physical_time, m_dt, m_multi_plasma);

        amrex::Real next_time = 0.;

        // adjust time step to reach max_time
        if (m_physical_time == m_max_time) {
            m_has_last_step = true;
            m_dt = 0.;
            next_time = 2. * m_max_time + 1.;
        } else if (m_physical_time + m_dt >= m_max_time) {
            m_dt = m_max_time - m_physical_time;
            next_time = m_max_time;
        } else {
            next_time = m_physical_time + m_dt;
        }

        if (m_verbose >= 1) {
            std::cout << "Rank " << rank
                      << " started  step " << step
                      << " at time = " << m_physical_time
                      << " with dt = " << m_dt << std::endl;
        }

        if (step+1 <= m_max_step) {
            m_multi_buffer.put_time(next_time);
        }

        // Only reset plasma after receiving time step, to use proper density
        m_multi_plasma.InitData(m_slice_ba, m_slice_dm, m_slice_geom, m_3D_geom);

        // deposit neutralizing background on every MR level
        if (m_N_level > 1) {
            m_multi_plasma.TagByLevel(m_N_level, m_3D_geom);
        }

        for (int lev=0; lev<m_N_level; ++lev) {
            if (m_do_tiling) {
                m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);
            }
            // Store charge density of (immobile) ions into WhichSlice::RhomJzIons
            m_multi_plasma.DepositNeutralizingBackground(
                m_fields, m_multi_laser, WhichSlice::RhomJzIons, m_3D_geom, lev);
        }

        // need correct physical time for this
        InitDiagnostics(step);

        // Solve slices
        for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
            SolveOneSlice(isl, step);
        };

        if (m_physical_time < m_max_time) {
            m_adaptive_time_step.Calculate(
                m_physical_time, m_dt, m_multi_beam, m_multi_plasma, 0, false);
        }

        WriteDiagnostics(step);

        m_multi_beam.InSituWriteToFile(step, m_physical_time, m_3D_geom[0], m_max_step, m_max_time);
        m_multi_plasma.InSituWriteToFile(step, m_physical_time, m_3D_geom[0], m_max_step, m_max_time);

        if (!m_explicit) {
            // averaging predictor corrector loop diagnostics
            m_predcorr_avg_iterations /= bx.length(Direction::z);
            m_predcorr_avg_B_error /= bx.length(Direction::z);
            if (m_verbose >= 2) {
                amrex::AllPrint() << "Rank " << rank
                                  << ": avg. number of iterations " << m_predcorr_avg_iterations
                                  <<" avg. transverse B field error " << m_predcorr_avg_B_error
                                  << "\n";
            }
            m_predcorr_avg_iterations = 0.;
            m_predcorr_avg_B_error = 0.;
        }

        FlushDiagnostics();
    }
}

void
Hipace::SolveOneSlice (int islice, int step)
{
#ifdef AMREX_USE_MPI
    {
        // Call a MPI function so that the MPI implementation has a chance to
        // run tasks necessary to make progress with asynchronous communications.
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }
#endif
    HIPACE_PROFILE("Hipace::SolveOneSlice()");

    // Get this laser slice from the 3D array
    m_multi_laser.Copy(islice, false);

    int current_N_level = 1;

    for (int lev=1; lev<m_N_level; ++lev) {
        if (m_3D_geom[lev].Domain().smallEnd(Direction::z) <= islice &&
            m_3D_geom[lev].Domain().bigEnd(Direction::z) >= islice) {
            current_N_level = lev + 1;
        }
    }

    if (islice == m_3D_geom[0].Domain().bigEnd(2)) {
        m_multi_buffer.get_data(islice, m_multi_beam, WhichBeamSlice::This);
    }
    if (islice-1 >= m_3D_geom[0].Domain().smallEnd(2)) {
        m_multi_buffer.get_data(islice-1, m_multi_beam, WhichBeamSlice::Next);
    }

    m_multi_beam.InSituComputeDiags(step, islice, m_max_step, m_physical_time, m_max_time);
    m_multi_plasma.InSituComputeDiags(step, islice, m_max_step, m_physical_time, m_max_time);
    FillBeamDiagnostics(step);

    if (m_N_level > 1) {
        m_multi_beam.TagByLevel(current_N_level, m_3D_geom, WhichSlice::This);
        m_multi_beam.TagByLevel(current_N_level, m_3D_geom, WhichSlice::Next);
        m_multi_plasma.TagByLevel(current_N_level, m_3D_geom);
    }

    // reorder plasma before TileSort
    m_multi_plasma.ReorderParticles(islice);

    // prepare/initialize fields
    for (int lev=0; lev<current_N_level; ++lev) {
        m_fields.InitializeSlices(lev, islice, m_3D_geom);
    }

    // deposit current
    for (int lev=0; lev<current_N_level; ++lev) {
        // tiling used by plasma current deposition
        if (m_do_tiling) m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);

        if (m_explicit) {
            // deposit jx, jy, chi and rhomjz for all plasmas
            m_multi_plasma.DepositCurrent(m_fields, m_multi_laser, WhichSlice::This,
                true, false, m_deposit_rho, true, true, m_3D_geom, lev);

            // deposit jx_beam and jy_beam in the Next slice
            m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step,
                m_do_beam_jx_jy_deposition, false, false, WhichSlice::Next, WhichBeamSlice::Next);

            // deposit jz_beam and maybe rhomjz of the beam on This slice
            m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step,
                false, true, m_do_beam_jz_minus_rho, WhichSlice::This, WhichBeamSlice::This);
        } else {
            // deposit jx jy jz (maybe chi) and rhomjz
            m_multi_plasma.DepositCurrent(m_fields, m_multi_laser, WhichSlice::This,
                true, true, m_deposit_rho, m_use_laser, true, m_3D_geom, lev);

            // deposit jx jy jz and maybe rhomjz on This slice
            m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step,
                m_do_beam_jx_jy_deposition, true, m_do_beam_jz_minus_rho,
                WhichSlice::This, WhichBeamSlice::This);
        }
        // add neutralizing background
        m_fields.AddRhoIons(lev);

        // deposit grid current into jz_beam
        m_grid_current.DepositCurrentSlice(m_fields, m_3D_geom[lev], lev, islice);
    }

    // Psi ExmBy EypBx Ez Bz solve
    m_fields.SolvePoissonPsiExmByEypBxEzBz(m_3D_geom, current_N_level);

    // Bx By solve
    if (m_explicit) {
        for (int lev=0; lev<current_N_level; ++lev) {
            // The algorithm used was derived in
            // [Wang, T. et al. Phys. Rev. Accel. Beams 25, 104603 (2022)],
            // it is implemented in the WAND-PIC quasistatic PIC code.

            // Set Sx and Sy to beam contribution
            InitializeSxSyWithBeam(lev);

            // Deposit Sx and Sy for every plasma species
            m_multi_plasma.ExplicitDeposition(m_fields, m_multi_laser, m_3D_geom, lev);

            // Solves Bx, By using Sx, Sy and chi
            ExplicitMGSolveBxBy(lev, WhichSlice::This);
        }
    } else {
        // Solves Bx and By in the current slice and modifies the force terms of the plasma particles
        PredictorCorrectorLoopToSolveBxBy(islice, current_N_level, step);
    }

    if (m_multi_beam.isSalameNow(step)) {
        // Modify the beam particle weights on this slice to flatten Ez.
        // As the beam current is modified, Bx and By are also recomputed.
        SalameModule(this, m_salame_n_iter, m_salame_do_advance, m_salame_last_slice,
                    m_salame_overloaded, current_N_level, step, islice);
    }

    // copy fields to diagnostic array
    for (int lev=0; lev<current_N_level; ++lev) {
        FillFieldDiagnostics(lev, islice);
    }

    // plasma ionization
    for (int lev=0; lev<current_N_level; ++lev) {
        m_multi_plasma.DoFieldIonization(lev, m_3D_geom[lev], m_fields);
    }

    // Push plasma particles
    for (int lev=0; lev<current_N_level; ++lev) {
        m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, m_3D_geom, false, lev);
    }

    // get minimum beam acceleration on level 0
    m_adaptive_time_step.GatherMinAccSlice(m_multi_beam,  m_3D_geom[0], m_fields);

    // Push beam particles
    m_multi_beam.AdvanceBeamParticlesSlice(m_fields, m_3D_geom, current_N_level,
                                           m_external_E_uniform, m_external_B_uniform,
                                           m_external_E_slope, m_external_B_slope);

    m_multi_beam.shiftSlippedParticles(islice, m_3D_geom[0]);

    bool is_last_step = (step == m_max_step) || (m_physical_time >= m_max_time);
    m_multi_buffer.put_data(islice, m_multi_beam, WhichBeamSlice::This, is_last_step);

    // collisions for all particles calculated on level 0
    m_multi_plasma.doCoulombCollision(0, m_slice_geom[0].Domain(), m_slice_geom[0]);

    // Advance laser slice by 1 step and store result to 3D array
    // no MR for laser
    m_multi_laser.AdvanceSlice(m_fields, m_3D_geom[0], m_dt, step);
    m_multi_laser.Copy(islice, true);

    // shift all levels
    for (int lev=0; lev<current_N_level; ++lev) {
        m_fields.ShiftSlices(lev);
    }

    m_multi_beam.shiftBeamSlices();
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
Hipace::InitializeSxSyWithBeam (const int lev)
{
    HIPACE_PROFILE("Hipace::InitializeSxSyWithBeam()");
    using namespace amrex::literals;

    if (!m_fields.m_extended_solve && lev==0) {
        if (m_explicit) {
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::Next,
                "jx_beam", "jy_beam");
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::This,
                "jz_beam");
        }
    }

    amrex::MultiFab& slicemf = m_fields.getSlices(lev);

    const amrex::Real dx = m_3D_geom[lev].CellSize(Direction::x);
    const amrex::Real dy = m_3D_geom[lev].CellSize(Direction::y);
    const amrex::Real dz = m_3D_geom[lev].CellSize(Direction::z);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        amrex::Box const& bx = mfi.tilebox();

        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Sx = Comps[WhichSlice::This]["Sx"];
        const int Sy = Comps[WhichSlice::This]["Sy"];
        const int next_jxb = Comps[WhichSlice::Next]["jx_beam"];
        const int next_jyb = Comps[WhichSlice::Next]["jy_beam"];
        const int      jzb = Comps[WhichSlice::This]["jz_beam"];
        const int prev_jxb = Comps[WhichSlice::Previous]["jx_beam"];
        const int prev_jyb = Comps[WhichSlice::Previous]["jy_beam"];

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

    // interpolate Sx, Sy and chi to lev from lev-1 in the domain edges.
    // This also accounts for jx_beam, jy_beam
    m_fields.LevelUpBoundary(m_3D_geom, lev, which_slice, "Sy",
        m_fields.m_poisson_nguards, -m_fields.m_slices_nguards);
    m_fields.LevelUpBoundary(m_3D_geom, lev, which_slice, "Sx",
        m_fields.m_poisson_nguards, -m_fields.m_slices_nguards);
    m_fields.LevelUpBoundary(m_3D_geom, lev, which_slice_chi, "chi",
        m_fields.m_poisson_nguards, -m_fields.m_slices_nguards);

    if (lev!=0 && (slicemf.box(0).length(0) % 2 == 0)) {
        // cell centered MG solve: no ghost cells, put boundary condition into source term
        // node centered MG solve: one ghost cell, use boundary condition from there
        m_fields.SetBoundaryCondition(m_3D_geom, lev, which_slice, "Bx",
                                      m_fields.getField(lev, which_slice, "Sy"));
        m_fields.SetBoundaryCondition(m_3D_geom, lev, which_slice, "By",
                                      m_fields.getField(lev, which_slice, "Sx"));
    }

    // interpolate Bx and By to lev from lev-1 in the ghost cells
    m_fields.LevelUpBoundary(m_3D_geom, lev, which_slice, "Bx",
        m_fields.m_slices_nguards, m_fields.m_poisson_nguards);
    m_fields.LevelUpBoundary(m_3D_geom, lev, which_slice, "By",
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
Hipace::PredictorCorrectorLoopToSolveBxBy (const int islice, const int current_N_level,
                                           const int step)
{
    HIPACE_PROFILE("Hipace::PredictorCorrectorLoopToSolveBxBy()");

    amrex::Real relative_Bfield_error_prev_iter = 1.0;
    amrex::Real relative_Bfield_error = m_fields.ComputeRelBFieldError(
        WhichSlice::Previous, WhichSlice::PCPrevIter, m_3D_geom, current_N_level);

    // Guess Bx and By on WhichSlice::This
    for (int lev=0; lev<current_N_level; ++lev) {
        m_fields.InitialBfieldGuess(relative_Bfield_error, m_predcorr_B_error_tolerance, lev);
    }

    for (int lev=0; lev<current_N_level; ++lev) {
        if (!m_fields.m_extended_solve && lev==0) {
            // exchange ExmBy EypBx Ez Bz
            m_fields.FillBoundary(m_3D_geom[lev].periodicity(), lev, WhichSlice::This,
                "ExmBy", "EypBx", "Ez", "Bz");
        }
        m_fields.setVal(0., lev, WhichSlice::PCIter, "Bx", "By");
        m_fields.duplicate(lev, WhichSlice::PCPrevIter, {"Bx", "By"},
                                WhichSlice::This,       {"Bx", "By"});
    }

    // Begin of predictor corrector loop
    int i_iter = 0;
    // resetting the initial B-field error for mixing between iterations
    relative_Bfield_error = 1.0;
    while (( relative_Bfield_error > m_predcorr_B_error_tolerance )
           && ( i_iter < m_predcorr_max_iterations ))
    {
        i_iter++;
        m_predcorr_avg_iterations += 1.0;

        for (int lev=0; lev<current_N_level; ++lev) {
            // Push particles to the next temp slice
            m_multi_plasma.AdvanceParticles(m_fields, m_multi_laser, m_3D_geom, true, lev);
        }

        if (m_N_level > 1) {
            // tag to temp slice for deposition
            m_multi_plasma.TagByLevel(current_N_level, m_3D_geom);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            if (m_do_tiling) m_multi_plasma.TileSort(m_slice_geom[lev].Domain(), m_slice_geom[lev]);
            // plasmas deposit jx jy to next temp slice
            m_multi_plasma.DepositCurrent(m_fields, m_multi_laser, WhichSlice::Next,
                true, false, false, false, false, m_3D_geom, lev);

            // beams deposit jx jy to the next slice
            m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, lev, step,
                m_do_beam_jx_jy_deposition, false, false, WhichSlice::Next, WhichBeamSlice::Next);
        }

        // Calculate Bx and By
        m_fields.SolvePoissonBxBy(m_3D_geom, current_N_level, WhichSlice::PCIter);

        relative_Bfield_error = m_fields.ComputeRelBFieldError(
            WhichSlice::This, WhichSlice::PCIter, m_3D_geom, current_N_level);

        if (i_iter == 1) relative_Bfield_error_prev_iter = relative_Bfield_error;

        for (int lev=0; lev<current_N_level; ++lev) {
            // Mixing the calculated B fields to the actual B field and shifting iterated B fields
            m_fields.MixAndShiftBfields(relative_Bfield_error, relative_Bfield_error_prev_iter,
                                        m_predcorr_B_mixing_factor, lev);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            // resetting current in the next slice to clean temporarily used current
            m_fields.setVal(0., lev, WhichSlice::Next, "jx", "jy");
        }

        if (m_N_level > 1) {
            // tag to prev for next push
            m_multi_plasma.TagByLevel(current_N_level, m_3D_geom, true);
        }

        // Shift relative_Bfield_error values
        relative_Bfield_error_prev_iter = relative_Bfield_error;
    } // end of predictor corrector loop

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
    if (m_verbose >= 2) amrex::Print() << "islice: " << islice <<
                " n_iter: "<<i_iter<<" relative B field error: "<<relative_Bfield_error<< "\n";
}

void
Hipace::InitDiagnostics (const int step)
{
#ifdef HIPACE_USE_OPENPMD
    // need correct physical time for this check
    if (m_diags.hasAnyOutput(step, m_max_step, m_physical_time, m_max_time)) {
        m_openpmd_writer.InitDiagnostics();
    }
    if (m_diags.hasBeamOutput(step, m_max_step, m_physical_time, m_max_time)) {
        m_openpmd_writer.InitBeamData(m_multi_beam, getDiagBeamNames());
    }
#endif
    for (int lev=0; lev<m_N_level; ++lev) {
        m_diags.ResizeFDiagFAB(m_3D_geom[lev].Domain(), lev, m_3D_geom[lev],
                               step, m_max_step, m_physical_time, m_max_time);
    }
}

void
Hipace::FillFieldDiagnostics (const int lev, int islice)
{
    for (auto& fd : m_diags.getFieldData()) {
        if (fd.m_level == lev && fd.m_has_field) {
            m_fields.Copy(lev, islice, fd.m_geom_io, fd.m_F,
                fd.m_F.box(), m_3D_geom[lev],
                fd.m_comps_output_idx, fd.m_nfields,
                fd.m_do_laser, m_multi_laser);
        }
    }
}

void
Hipace::FillBeamDiagnostics (const int step)
{
#ifdef HIPACE_USE_OPENPMD
    if (m_diags.hasBeamOutput(step, m_max_step, m_physical_time, m_max_time)) {
        m_openpmd_writer.CopyBeams(m_multi_beam, getDiagBeamNames());
    }
#else
    amrex::ignore_unused(step);
#endif
}

void
Hipace::WriteDiagnostics (const int step)
{
#ifdef HIPACE_USE_OPENPMD
    if (m_diags.hasAnyFieldOutput(step, m_max_step, m_physical_time, m_max_time)) {
        m_openpmd_writer.WriteDiagnostics(m_diags.getFieldData(), m_multi_beam,
                        m_physical_time, step, getDiagBeamNames(),
                        m_3D_geom, OpenPMDWriterCallType::fields);
    }

    if (m_diags.hasBeamOutput(step, m_max_step, m_physical_time, m_max_time)) {
        m_openpmd_writer.WriteDiagnostics(m_diags.getFieldData(), m_multi_beam,
                        m_physical_time, step, getDiagBeamNames(),
                        m_3D_geom, OpenPMDWriterCallType::beams);
    }
#else
    amrex::ignore_unused(step);
    amrex::Print()<<"WARNING: HiPACE++ compiled without openPMD support, the simulation has no I/O.\n";
#endif
}

void
Hipace::FlushDiagnostics ()
{
#ifdef HIPACE_USE_OPENPMD
    m_openpmd_writer.flush();
#endif
}
