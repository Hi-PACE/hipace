#include "Hipace.H"
#include "particles/deposition/BeamDepositCurrent.H"
#include "particles/deposition/PlasmaDepositCurrent.H"
#include "utils/HipaceProfilerWrapper.H"
#include "particles/pusher/PlasmaParticleAdvance.H"
#include "particles/pusher/BeamParticleAdvance.H"
#include "particles/BinSort.H"
#include "utils/IOUtil.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>

#include <algorithm>
#include <memory>

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 1000;
    constexpr int pcomm_z_tag = 1001;
}
#endif

Hipace* Hipace::m_instance = nullptr;

amrex::Real Hipace::m_dt = 0.0;
bool Hipace::m_normalized_units = false;
int Hipace::m_verbose = 0;
int Hipace::m_depos_order_xy = 2;
int Hipace::m_depos_order_z = 0;
amrex::Real Hipace::m_predcorr_B_error_tolerance = 4e-2;
int Hipace::m_predcorr_max_iterations = 30;
amrex::Real Hipace::m_predcorr_B_mixing_factor = 0.05;
bool Hipace::m_slice_beam = false;
bool Hipace::m_3d_on_host = false;
bool Hipace::m_do_device_synchronize = false;
bool Hipace::m_slice_F_xz = false;
bool Hipace::m_output_plasma = false;
int Hipace::m_beam_injection_cr = 1;
amrex::Real Hipace::m_external_focusing_field_strength = 0.;
amrex::Real Hipace::m_external_accel_field_strength = 0.;

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
    pph.query("output_slice", m_slice_F_xz);
    pph.query("beam_injection_cr", m_beam_injection_cr);
    pph.query("slice_beam", m_slice_beam);
    pph.query("3d_on_host", m_3d_on_host);
    if (m_3d_on_host) AMREX_ALWAYS_ASSERT(m_slice_beam);
    if (m_slice_F_xz) AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_slice_beam,
                        "To enable slice I/O, 'hipace.slice_beam = 1' is required ");
    m_numprocs_z = amrex::ParallelDescriptor::NProcs() / (m_numprocs_x*m_numprocs_y);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs_x*m_numprocs_y*m_numprocs_z
                                     == amrex::ParallelDescriptor::NProcs(),
                                     "Check hipace.numprocs_x and hipace.numprocs_y");
    pph.query("do_device_synchronize", m_do_device_synchronize);
    pph.query("output_plasma", m_output_plasma);
    pph.query("external_focusing_field_strength", m_external_focusing_field_strength);
    pph.query("external_accel_field_strength", m_external_accel_field_strength);

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
    m_plasma_container.SetParticleBoxArray(lev, m_slice_ba);
    m_plasma_container.SetParticleDistributionMap(lev, m_slice_dm);
    m_plasma_container.SetParticleGeometry(lev, m_slice_geom);
    m_plasma_container.InitData();
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
    InitDiagnostics();
    WriteDiagnostics(0);
    for (int step = 0; step < m_max_step; ++step)
    {
        /* calculate the adaptive time step before printout, so the ranks already print their new dt */
        m_adaptive_time_step.Calculate(m_dt, step, m_multi_beam, m_plasma_container, lev, m_comm_z);

        if (m_verbose>=1) std::cout<<"Rank "<<rank<<" started  step "<<step<<" with dt = "<<m_dt<<'\n';

        ResetAllQuantities(lev);

        Wait();

        amrex::MultiFab& fields = m_fields.getF(lev);

        if (!m_slice_beam) m_multi_beam.DepositCurrent(m_fields, geom[lev], lev);

        /* Store charge density of (immobile) ions into WhichSlice::RhoIons */
        if (m_rank_z == m_numprocs_z-1){
            DepositCurrent(m_plasma_container, m_fields, WhichSlice::RhoIons,
                           false, false, false, true, geom[lev], lev);
        }

        // Loop over longitudinal boxes on this rank, from head to tail
        const amrex::Vector<int> index_array = fields.IndexArray();
        for (auto it = index_array.rbegin(); it != index_array.rend(); ++it)
        {
            const amrex::Box& bx = fields.box(*it);
            amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>> bins;
            if (m_slice_beam) bins = m_multi_beam.findParticlesInEachSlice(lev, *it, bx, geom[lev]);

            for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
                SolveOneSlice(isl, lev, bins);
            };
        }
        if (amrex::ParallelDescriptor::NProcs() == 1) {
            m_multi_beam.Redistribute();
        } else {
            amrex::Print()<<"WARNING: In parallel runs, beam particles are not redistributed. \n";
        }

        /* Passing the adaptive time step info */
        m_adaptive_time_step.PassTimeStepInfo(step, m_comm_z);
        // Slices have already been shifted, so send
        // slices {2,3} from upstream to {2,3} in downstream.
        Notify();

#ifdef HIPACE_USE_OPENPMD
        WriteDiagnostics(step+1);
#else
        amrex::Print()<<"WARNING: In parallel runs, only openPMD supports dumping all time steps. \n";
#endif
        m_physical_time += m_dt;
    }
    // For consistency, decrement the physical time, so the last time step is like the others:
    // the time stored in the output file is the time for the fields. The beam is one time step
    // ahead.
    m_physical_time -= m_dt;
    WriteDiagnostics(m_max_step, true);
#ifdef HIPACE_USE_OPENPMD
    m_outputSeries.reset();
#endif
}

void
Hipace::SolveOneSlice (int islice, int lev, amrex::Vector<amrex::DenseBins<BeamParticleContainer::ParticleType>>& bins)
{
    // Between this push and the corresponding pop at the end of this
    // for loop, the parallelcontext is the transverse communicator
    amrex::ParallelContext::push(m_comm_xy);

    if (m_slice_beam){
        m_fields.getSlices(lev, WhichSlice::This).setVal(0.);
    } else {
        m_fields.Copy(lev, islice, FieldCopyType::FtoS, 0, 0, FieldComps::nfields);
    }

    AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                           WhichSlice::This, false,
                           true, false, false, lev);

    m_plasma_container.RedistributeSlice(lev);
    amrex::MultiFab rho(m_fields.getSlices(lev, WhichSlice::This), amrex::make_alias,
                        FieldComps::rho, 1);

    DepositCurrent(m_plasma_container, m_fields, WhichSlice::This, false, true,
                   true, true, geom[lev], lev);
    m_fields.AddRhoIons(lev);

    // need to exchange jx jy jz rho
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        FieldComps::jy == FieldComps::jx+1 && FieldComps::jz == FieldComps::jx+2 &&
        FieldComps::rho == FieldComps::jx+3, "The order of jx, jy, jz, rho must not be "
        "changed, because the 4 components starting from jx are grabbed at once");
    amrex::MultiFab j_slice(m_fields.getSlices(lev, WhichSlice::This),
                            amrex::make_alias, FieldComps::jx, 4);
    j_slice.FillBoundary(Geom(lev).periodicity());

    m_fields.SolvePoissonExmByAndEypBx(Geom(lev), m_comm_xy, lev);

    if (m_slice_beam) m_multi_beam.DepositCurrentSlice(
        m_fields, geom[lev], lev, islice, bins);

    j_slice.FillBoundary(Geom(lev).periodicity());

    m_fields.SolvePoissonEz(Geom(lev),lev);
    m_fields.SolvePoissonBz(Geom(lev), lev);

    /* Modifies Bx and By in the current slice
     * and the force terms of the plasma particles
     */
    PredictorCorrectorLoopToSolveBxBy(islice, lev);

    // Push beam particles
    if (m_slice_beam) m_multi_beam.AdvanceBeamParticlesSlice(
        m_fields, geom[lev], lev, islice, bins);

    m_fields.Copy(lev, islice, FieldCopyType::StoF, 0, 0, FieldComps::nfields);

    m_fields.ShiftSlices(lev);

    // After this, the parallel context is the full 3D communicator again
    amrex::ParallelContext::pop();
}

void
Hipace::ResetAllQuantities (int lev)
{
    HIPACE_PROFILE("Hipace::ResetAllQuantities()");
    ResetPlasmaParticles(m_plasma_container, lev, true);

    amrex::MultiFab& fields = m_fields.getF(lev);
    for (int islice=0; islice<(int) WhichSlice::N; islice++) {
        m_fields.getSlices(lev, islice).setVal(0.);
    }

    if (!m_slice_beam) fields.setVal(0.);
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
        FieldComps::Bx, FieldComps::By,FieldComps::Bx, FieldComps::By, Geom(lev), lev);

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
                          FieldComps::Bx, 0, 1, 0);
    amrex::MultiFab By_prev_iter(m_fields.getSlices(lev, WhichSlice::This).boxArray(),
                                 m_fields.getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                                 m_fields.getSlices(lev, WhichSlice::This).nGrowVect());
    amrex::MultiFab::Copy(By_prev_iter, m_fields.getSlices(lev, WhichSlice::This),
                          FieldComps::By, 0, 1, 0);

    /* creating aliases to the current in the next slice.
     * This needs to be reset after each push to the next slice */
    amrex::MultiFab jx_next(m_fields.getSlices(lev, WhichSlice::Next),
                            amrex::make_alias, FieldComps::jx, 1);
    amrex::MultiFab jy_next(m_fields.getSlices(lev, WhichSlice::Next),
                            amrex::make_alias, FieldComps::jy, 1);


    /* shift force terms, update force terms using guessed Bx and By */
    AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                           WhichSlice::This, false,
                           false, true, true, lev);

    /* Begin of predictor corrector loop  */
    int i_iter = 0;
    /* resetting the initial B-field error for mixing between iterations */
    relative_Bfield_error = 1.0;
    while (( relative_Bfield_error > m_predcorr_B_error_tolerance )
           && ( i_iter < m_predcorr_max_iterations ))
    {
        i_iter++;
        /* Push particles to the next slice */
        AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                               WhichSlice::Next, true,
                               true, false, false, lev);
        m_plasma_container.RedistributeSlice(lev);

        /* deposit current to next slice */
        DepositCurrent(m_plasma_container, m_fields, WhichSlice::Next, true,
                       true, false, false, geom[lev], lev);
        amrex::ParallelContext::push(m_comm_xy);
        // need to exchange jx jy jz rho
        amrex::MultiFab j_slice_next(m_fields.getSlices(lev, WhichSlice::Next),
                                     amrex::make_alias, FieldComps::jx, 4);
        j_slice_next.FillBoundary(Geom(lev).periodicity());
        amrex::ParallelContext::pop();

        /* Calculate Bx and By */
        m_fields.SolvePoissonBx(Bx_iter, Geom(lev), lev);
        m_fields.SolvePoissonBy(By_iter, Geom(lev), lev);

        relative_Bfield_error = m_fields.ComputeRelBFieldError(
                                               m_fields.getSlices(lev, WhichSlice::This),
                                               m_fields.getSlices(lev, WhichSlice::This),
                                               Bx_iter, By_iter, FieldComps::Bx,
                                               FieldComps::By, 0, 0, Geom(lev), lev);

        if (i_iter == 1) relative_Bfield_error_prev_iter = relative_Bfield_error;

        /* Mixing the calculated B fields to the actual B field and shifting iterated B fields */
        m_fields.MixAndShiftBfields(Bx_iter, Bx_prev_iter, FieldComps::Bx, relative_Bfield_error,
                                    relative_Bfield_error_prev_iter, m_predcorr_B_mixing_factor,
                                    lev);
        m_fields.MixAndShiftBfields(By_iter, By_prev_iter, FieldComps::By, relative_Bfield_error,
                                    relative_Bfield_error_prev_iter, m_predcorr_B_mixing_factor,
                                    lev);

        /* resetting current in the next slice to clean temporarily used current*/
        jx_next.setVal(0.);
        jy_next.setVal(0.);

        amrex::ParallelContext::push(m_comm_xy);
         // exchange Bx By
        m_fields.getSlices(lev, WhichSlice::This).FillBoundary(Geom(lev).periodicity());
        amrex::ParallelContext::pop();

        /* Update force terms using the calculated Bx and By */
        AdvancePlasmaParticles(m_plasma_container, m_fields, geom[lev],
                               WhichSlice::Next, false,
                               false, true, false, lev);

        /* Shift relative_Bfield_error values */
        relative_Bfield_error_prev_iter = relative_Bfield_error;
    } /* end of predictor corrector loop */

    /* resetting the particle position after they have been pushed to the next slice */
    ResetPlasmaParticles(m_plasma_container, lev);

    if (relative_Bfield_error > 10. && m_predcorr_B_error_tolerance > 0.)
    {
        amrex::Abort("Predictor corrector loop diverged!\n"
                     "Re-try by adjusting the following paramters in the input script:\n"
                     "- lower mixing factor: hipace.predcorr_B_mixing_factor "
                     "(hidden default: 0.1) \n"
                     "- lower B field error tolerance: hipace.predcorr_B_error_tolerance"
                     " (hidden default: 0.04)\n"
                     "- higher number of iterations in the pred. cor. loop:"
                     "hipace.predcorr_max_iterations (hidden default: 5)\n"
                     "- higher longitudinal resolution");
    }
    if (m_verbose >= 2) amrex::Print()<<"islice: " << islice << " n_iter: "<<i_iter<<
                            " relative B field error: "<<relative_Bfield_error<< "\n";
}

void
Hipace::Wait ()
{
    HIPACE_PROFILE("Hipace::Wait()");
#ifdef AMREX_USE_MPI
    if (m_rank_z != m_numprocs_z-1) {
        {
        const int lev = 0;
        amrex::MultiFab& slice2 = m_fields.getSlices(lev, WhichSlice::Previous1);
        amrex::MultiFab& slice3 = m_fields.getSlices(lev, WhichSlice::Previous2);
        amrex::MultiFab& slice4 = m_fields.getSlices(lev, WhichSlice::RhoIons);
        // Note that there is only one local Box in slice multifab's boxarray.
        const int box_index = slice2.IndexArray()[0];
        amrex::Array4<amrex::Real> const& slice_fab2 = slice2.array(box_index);
        amrex::Array4<amrex::Real> const& slice_fab3 = slice3.array(box_index);
        amrex::Array4<amrex::Real> const& slice_fab4 = slice4.array(box_index);
        const amrex::Box& bx = slice2.boxArray()[box_index]; // does not include ghost cells
        const std::size_t nreals_valid_slice2 = bx.numPts()*slice_fab2.nComp();
        const std::size_t nreals_valid_slice3 = bx.numPts()*slice_fab3.nComp();
        const std::size_t nreals_valid_slice4 = bx.numPts()*slice_fab4.nComp();
        const std::size_t nreals_total =
            nreals_valid_slice2 + nreals_valid_slice3 + nreals_valid_slice4;
        auto recv_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real)*nreals_total);
        auto const buf2 = amrex::makeArray4(recv_buffer,
                                            bx, slice_fab2.nComp());
        auto const buf3 = amrex::makeArray4(recv_buffer+nreals_valid_slice2,
                                            bx, slice_fab3.nComp());
        auto const buf4 = amrex::makeArray4(recv_buffer+nreals_valid_slice2+nreals_valid_slice3,
                                            bx, slice_fab4.nComp());
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
             },
             bx, slice_fab4.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 slice_fab4(i,j,k,n) = buf4(i,j,k,n);
             });

        amrex::Gpu::Device::synchronize();
        amrex::The_Pinned_Arena()->free(recv_buffer);
        }

        // Same thing for the plasma particles. Currently only one tile.
        {
            const int lev = 0;
            amrex::Long np = m_plasma_container.m_num_exchange;
            amrex::Long psize = m_plasma_container.superParticleSize();
            amrex::Long buffer_size = psize*np;
            auto recv_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

            MPI_Status status;
            MPI_Recv(recv_buffer, buffer_size,
                     amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                     m_rank_z+1, pcomm_z_tag, m_comm_z, &status);

            auto& ptile = m_plasma_container.DefineAndReturnParticleTile(lev, 0, 0);
            ptile.resize(np);
            auto ptd = ptile.getParticleTileData();

            amrex::Gpu::DeviceVector<int> comm_real(m_plasma_container.NumRealComps(), 1);
            amrex::Gpu::DeviceVector<int> comm_int (m_plasma_container.NumIntComps(),  1);
            auto p_comm_real = comm_real.data();
            auto p_comm_int = comm_int.data();
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                ptd.unpackParticleData(recv_buffer, i*psize, i, p_comm_real, p_comm_int);
            });

            amrex::Gpu::Device::synchronize();
            amrex::The_Pinned_Arena()->free(recv_buffer);
        }
    }
    #ifdef HIPACE_USE_OPENPMD
    m_multi_beam.WaitNumParticles(m_comm_z);
    #endif
#endif
}

void
Hipace::Notify ()
{
    HIPACE_PROFILE("Hipace::Notify()");
    // Send from slices 2 and 3 (or main MultiFab's first two valid slabs) to receiver's slices 2
    // and 3.

#ifdef AMREX_USE_MPI
    if (m_rank_z != 0) {
        NotifyFinish(); // finish the previous send

        {
        const int lev = 0;
        const amrex::MultiFab& slice2 = m_fields.getSlices(lev, WhichSlice::Previous1);
        const amrex::MultiFab& slice3 = m_fields.getSlices(lev, WhichSlice::Previous2);
        const amrex::MultiFab& slice4 = m_fields.getSlices(lev, WhichSlice::RhoIons);
        // Note that there is only one local Box in slice multifab's boxarray.
        const int box_index = slice2.IndexArray()[0];
        amrex::Array4<amrex::Real const> const& slice_fab2 = slice2.array(box_index);
        amrex::Array4<amrex::Real const> const& slice_fab3 = slice3.array(box_index);
        amrex::Array4<amrex::Real const> const& slice_fab4 = slice4.array(box_index);
        const amrex::Box& bx = slice2.boxArray()[box_index]; // does not include ghost cells
        const std::size_t nreals_valid_slice2 = bx.numPts()*slice_fab2.nComp();
        const std::size_t nreals_valid_slice3 = bx.numPts()*slice_fab3.nComp();
        const std::size_t nreals_valid_slice4 = bx.numPts()*slice_fab4.nComp();
        const std::size_t nreals_total =
            nreals_valid_slice2 + nreals_valid_slice3 + nreals_valid_slice4;
        m_send_buffer = (amrex::Real*)amrex::The_Pinned_Arena()->alloc
            (sizeof(amrex::Real)*nreals_total);
        auto const buf2 = amrex::makeArray4(m_send_buffer,
                                            bx, slice_fab2.nComp());
        auto const buf3 = amrex::makeArray4(m_send_buffer+nreals_valid_slice2,
                                            bx, slice_fab3.nComp());
        auto const buf4 = amrex::makeArray4(m_send_buffer+nreals_valid_slice2+nreals_valid_slice3,
                                            bx, slice_fab4.nComp());
        amrex::ParallelFor
            (bx, slice_fab2.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 buf2(i,j,k,n) = slice_fab2(i,j,k,n);
             },
             bx, slice_fab3.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 buf3(i,j,k,n) = slice_fab3(i,j,k,n);
             },
             bx, slice_fab4.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
             {
                 buf4(i,j,k,n) = slice_fab4(i,j,k,n);
             });

        amrex::Gpu::Device::synchronize();
        MPI_Isend(m_send_buffer, nreals_total,
                  amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                  m_rank_z-1, comm_z_tag, m_comm_z, &m_send_request);
        }

        // Same thing for the plasma particles. Currently only one tile.
        {
            const int lev = 0;

            amrex::Long np = m_plasma_container.m_num_exchange;
            amrex::Long psize = m_plasma_container.superParticleSize();
            amrex::Long buffer_size = psize*np;
            m_psend_buffer = (char*)amrex::The_Pinned_Arena()->alloc(buffer_size);

            auto& ptile = m_plasma_container.ParticlesAt(lev, 0, 0);
            const auto ptd = ptile.getConstParticleTileData();

            amrex::Gpu::DeviceVector<int> comm_real(m_plasma_container.NumRealComps(), 1);
            amrex::Gpu::DeviceVector<int> comm_int (m_plasma_container.NumIntComps(),  1);
            auto p_comm_real = comm_real.data();
            auto p_comm_int = comm_int.data();
            auto p_psend_buffer = m_psend_buffer;
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                ptd.packParticleData(p_psend_buffer, i, i*psize, p_comm_real, p_comm_int);
            });

            amrex::Gpu::Device::synchronize();
            MPI_Isend(m_psend_buffer, buffer_size,
                      amrex::ParallelDescriptor::Mpi_typemap<char>::type(),
                      m_rank_z-1, pcomm_z_tag, m_comm_z, &m_psend_request);
        }
    }
    #ifdef HIPACE_USE_OPENPMD
    /* pass the number of beam particles from the upstream ranks to get the offset for openPMD IO */
    m_multi_beam.NotifyNumParticles(m_comm_z);
    #endif
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
        if (m_send_buffer) {
            MPI_Status status;
            MPI_Wait(&m_psend_request, &status);
            amrex::The_Pinned_Arena()->free(m_psend_buffer);
            m_psend_buffer = nullptr;
        }
    }
#endif
}

void
Hipace::WriteDiagnostics (int output_step, bool force_output)
{
    HIPACE_PROFILE("Hipace::WriteDiagnostics()");

    // Dump before first and after last step, and every m_output_period steps in-between
    if (m_output_period < 0 ||
        (!force_output && output_step % m_output_period != 0) ||
        output_step == m_last_output_dumped) return;

    // Store this dump iteration
    m_last_output_dumped = output_step;

    // Write fields
    const std::string filename = amrex::Concatenate("plt", output_step);
    // assumption: same order as in struct enum Field Comps
    const amrex::Vector< std::string > varnames
        {"ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "rho", "Psi"};

    amrex::Vector<std::string> rfs;
    amrex::Vector<amrex::Geometry> geom_io = Geom();

    constexpr int lev = 0; // we only have one level
    constexpr int idim = 1;
    amrex::RealBox prob_domain = Geom(lev).ProbDomain();
    amrex::Box domain = Geom(lev).Domain();
    // Define slice box
    int const icenter = domain.length(idim)/2;
    domain.setSmall(idim, icenter);
    domain.setBig(idim, icenter);
    if (m_slice_F_xz){
        geom_io[lev] = amrex::Geometry(domain, &prob_domain, Geom(lev).Coord());
    }

#ifdef HIPACE_USE_OPENPMD
    io::Iteration iteration = m_outputSeries->iterations[output_step];
    iteration.setTime(m_physical_time);

    // todo: periodicity/boundary, field solver, particle pusher, etc.
    auto meshes = iteration.meshes;


    // loop over field components
    for (int icomp = 0; icomp < FieldComps::nfields; ++icomp)
    {
        auto const& mf = m_fields.getF(lev);

        std::string fieldname = varnames[icomp];
        //                      "B"                "x" (todo)
        //                      "Bx"               ""  (just for now)
        io::Mesh field = meshes[fieldname];
        io::MeshRecordComponent field_comp = field[io::MeshRecordComponent::SCALAR];

        // meta-data
        field.setDataOrder(io::Mesh::DataOrder::C);
        //   node staggering
        auto relative_cell_pos = utils::getRelativeCellPosition(mf);      // AMReX Fortran index order
        std::reverse(relative_cell_pos.begin(), relative_cell_pos.end()); // now in C order
        //   labels, spacing and offsets
        std::vector< std::string > axisLabels {"z", "y", "x"};
        auto dCells = utils::getReversedVec(geom_io[lev].CellSize()); // dx, dy, dz
        auto offWindow = utils::getReversedVec(geom_io[lev].ProbLo()); // start of moving window
        if (m_slice_F_xz) {
            relative_cell_pos.erase(relative_cell_pos.begin() + 1);  // remove for y
            axisLabels.erase(axisLabels.begin() + 1); // remove y
            dCells.erase(dCells.begin() + 1); // remove dy
            offWindow.erase(offWindow.begin() + 1); // remove offset in y
        }
        field_comp.setPosition(relative_cell_pos);
        field.setAxisLabels(axisLabels);
        field.setGridSpacing(dCells);
        field.setGridGlobalOffset(offWindow);

        // data type and global size of the simulation
        io::Datatype datatype = io::determineDatatype< amrex::Real >();
        io::Extent global_size = utils::getReversedVec(geom_io[lev].Domain().size());
        if (m_slice_F_xz) global_size.erase(global_size.begin() + 1);  // remove Ny

        io::Dataset dataset(datatype, global_size);
        field_comp.resetDataset(dataset);

        // Loop over longitudinal boxes on this rank, from head to tail:
        // Loop through the multifab and store each box as a chunk with openpmd
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
        {
            amrex::FArrayBox const& fab = mf[mfi]; // note: this might include guards
            amrex::Box const data_box = mfi.validbox();  // w/o guards in all cases
            std::shared_ptr< amrex::Real const > data;
            if (mfi.validbox() == fab.box() ) {
                data = io::shareRaw( fab.dataPtr( icomp ) ); // non-owning view until flush()
            } else {
                // copy data to cut away guards
                amrex::FArrayBox io_fab(mfi.validbox(), 1, amrex::The_Pinned_Arena());
                io_fab.copy< amrex::RunOn::Host >(fab, fab.box(), icomp, mfi.validbox(), 0, 1);
                // move ownership into a shared pointer: openPMD-api will free the copy in flush()
                data = std::move(io_fab.release()); // note: a move honors the custom array-deleter
            }

            // Determine the offset and size of this data chunk in the global output
            amrex::IntVect const box_offset = data_box.smallEnd();
            io::Offset chunk_offset = utils::getReversedVec(box_offset);
            io::Extent chunk_size = utils::getReversedVec(data_box.size());
            if (m_slice_F_xz) { // remove Ny components
                chunk_offset.erase(chunk_offset.begin() + 1);
                chunk_size.erase(chunk_size.begin() + 1);
            }

            field_comp.storeChunk(data, chunk_offset, chunk_size);
        }
    }
    m_outputSeries->flush();
#endif

#ifndef HIPACE_USE_OPENPMD
    constexpr int nlev = 1;
    const amrex::IntVect local_ref_ratio {1, 1, 1};

    amrex::WriteMultiLevelPlotfile(
        filename, nlev, amrex::GetVecOfConstPtrs(m_fields.getF()), varnames,
        geom_io, m_physical_time, {output_step}, {local_ref_ratio},
        "HyperCLaw-V1.1", "Level_", "Cell", rfs);

    // Write beam particles
    m_multi_beam.WritePlotFile(filename);

    // Write plasma particles
    if (m_output_plasma){
        amrex::ParallelContext::push(m_comm_xy);
        m_plasma_container.Redistribute();
        amrex::ParallelContext::pop();
        amrex::Vector<int> plot_flags(PlasmaIdx::nattribs, 1);
        amrex::Vector<int> int_flags(PlasmaIdx::nattribs, 1);
        amrex::Vector<std::string> real_names {
            "w","ux","uy", "psi",
            "x_prev", "y_prev", "w_temp", "ux_temp", "uy_temp", "psi_temp",
            "Fx1", "Fx2", "Fx3", "Fx4", "Fx5",
            "Fy1", "Fy2", "Fy3", "Fy4", "Fy5",
            "Fux1", "Fux2", "Fux3", "Fux4", "Fux5",
            "Fuy1", "Fuy2", "Fuy3", "Fuy4", "Fuy5",
            "Fpsi1", "Fpsi2", "Fpsi3", "Fpsi4", "Fpsi5",
            "x0", "y0"
        };
        AMREX_ALWAYS_ASSERT(real_names.size() == PlasmaIdx::nattribs);
        amrex::Vector<std::string> int_names {};
        m_plasma_container.WritePlotFile(
            filename, "plasma",
            plot_flags, int_flags,
            real_names, int_names);
    }
#endif
}

void
Hipace::InitDiagnostics ()
{
    HIPACE_PROFILE("Hipace::InitDiagnostics()");

#ifdef HIPACE_USE_OPENPMD
    std::string filename = "diags/openPMD/openpmd.h5"; // bp or h5
#   ifdef AMREX_USE_MPI
    m_outputSeries = std::make_unique< io::Series >(
        filename, io::Access::CREATE, amrex::ParallelDescriptor::Communicator());
#   else
    m_outputSeries = std::make_unique< io::Series >(
        filename, io::Access::CREATE);
#   endif

    // open files early and collectively, so later flush calls are non-collective
    m_outputSeries->setIterationEncoding( io::IterationEncoding::groupBased );
    m_outputSeries->flush();

    // TODO: meta-data: author, mesh path, extensions, software
#endif
}
