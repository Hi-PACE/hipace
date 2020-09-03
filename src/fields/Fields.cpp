#include "Fields.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

Fields::Fields (Hipace const* a_hipace)
    : m_hipace(a_hipace),
      m_F(a_hipace->maxLevel()+1),
      m_slices(a_hipace->maxLevel()+1)
{}

void
Fields::AllocData (int lev, const amrex::BoxArray& ba,
                   const amrex::DistributionMapping& dm, amrex::Geometry const& geom)
{
    HIPACE_PROFILE("Fields::AllocData()");
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = std::max(1, Hipace::m_depos_order_xy);
    m_nguards = {nguards_xy, nguards_xy, Hipace::m_depos_order_z};
    m_slices_nguards = {nguards_xy, nguards_xy, 0};
    if (Hipace::m_3d_on_host){
        // The Arena uses pinned memory.
        m_F[lev].define(ba, dm, FieldComps::nfields, m_nguards,
                        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    } else {
        // The Arena uses managed memory.
        m_F[lev].define(ba, dm, FieldComps::nfields, m_nguards,
                        amrex::MFInfo().SetArena(amrex::The_Arena()));
    }

    std::map<int,amrex::Vector<amrex::Box> > boxes;
    for (int i = 0; i < ba.size(); ++i) {
        int rank = dm[i];
        if (m_hipace->InSameTransverseCommunicator(rank)) {
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

    amrex::BoxArray slice_ba(std::move(bl));
    amrex::DistributionMapping slice_dm(std::move(procmap));

    for (int islice=0; islice<(int) WhichSlice::N; islice++) {
        m_slices[lev][islice].define(slice_ba, slice_dm, FieldComps::nfields, m_slices_nguards,
                                     amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[lev][islice].setVal(0.0);
    }

    // The Poisson solver operates on transverse slices only.
    // The constructor takes the BoxArray and the DistributionMap of a slice,
    // so the FFTPlans are built on a slice.
    m_poisson_solver = FFTPoissonSolver(
        getSlices(lev, WhichSlice::This).boxArray(),
        getSlices(lev, WhichSlice::This).DistributionMap(),
        geom);
}

void
Fields::TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                              const amrex::Real dx, const amrex::Real mult_coeff,
                              const SliceOperatorType slice_operator,
                              const int scomp, const int dcomp)
{
    HIPACE_PROFILE("Fields::TransverseDerivative()");
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT((direction == Direction::x) || (direction == Direction::y));
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (direction == Direction::x){
                    /* finite difference along x */
                    if (slice_operator==SliceOperatorType::Assign)
                    {
                        dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dx) *
                          (src_array(i+1, j, k, scomp) - src_array(i-1, j, k, scomp));
                    }
                    else /* SliceOperatorType::Add */
                    {
                        dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dx) *
                          (src_array(i+1, j, k, scomp) - src_array(i-1, j, k, scomp));
                    }
                } else /* Direction::y */ {
                    /* finite difference along y */
                    if (slice_operator==SliceOperatorType::Assign)
                    {
                        dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dx) *
                          (src_array(i, j+1, k, scomp) - src_array(i, j-1, k, scomp));
                    }
                    else /* SliceOperatorType::Add */
                    {
                        dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dx) *
                          (src_array(i, j+1, k, scomp) - src_array(i, j-1, k, scomp));
                    }
                }
            }
            );
    }
}

void Fields::LongitudinalDerivative (const amrex::MultiFab& src1, const amrex::MultiFab& src2,
                             amrex::MultiFab& dst, const amrex::Real dz,
                             const amrex::Real mult_coeff,
                             const SliceOperatorType slice_operator,
                             const int s1comp, const int s2comp, const int dcomp)
{
    HIPACE_PROFILE("Fields::LongitudinalDerivative()");
    using namespace amrex::literals;
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src1_array = src1.array(mfi);
        amrex::Array4<amrex::Real const> const & src2_array = src2.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (slice_operator==SliceOperatorType::Assign)
                {
                    dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dz) *
                      (src1_array(i, j, k, s1comp) - src2_array(i, j, k, s2comp));
                }
                else /* SliceOperatorType::Add */
                {
                    dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dz) *
                      (src1_array(i, j, k, s1comp) - src2_array(i, j, k, s2comp));
                }

            }
            );
    }
}


void
Fields::Copy (int lev, int i_slice, FieldCopyType copy_type, int slice_comp, int full_comp,
              int ncomp)
{
    HIPACE_PROFILE("Fields::Copy()");
    auto& slice_mf = m_slices[lev][(int) WhichSlice::This];  // always slice #1
    amrex::Array4<amrex::Real> slice_array; // There is only one Box.
    for (amrex::MFIter mfi(slice_mf); mfi.isValid(); ++mfi) {
        auto& slice_fab = slice_mf[mfi];
        amrex::Box slice_box = slice_fab.box();
        slice_box.setSmall(Direction::z, i_slice);
        slice_box.setBig  (Direction::z, i_slice);
        slice_array = amrex::makeArray4(slice_fab.dataPtr(), slice_box, slice_fab.nComp());
        // slice_array's longitude index is i_slice.
    }

    auto& full_mf = m_F[lev];
    for (amrex::MFIter mfi(full_mf); mfi.isValid(); ++mfi) {
        amrex::Box const& vbx = mfi.validbox();
        if (vbx.smallEnd(Direction::z) <= i_slice and
            vbx.bigEnd  (Direction::z) >= i_slice)
        {
            amrex::Box copy_box = amrex::grow(vbx, m_slices_nguards);
            copy_box.setSmall(Direction::z, i_slice);
            copy_box.setBig  (Direction::z, i_slice);
            auto const& full_array = full_mf.array(mfi);
            if (copy_type == FieldCopyType::FtoS) {
                amrex::ParallelFor(copy_box, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    slice_array(i,j,k,n+slice_comp) = full_array(i,j,k,n+full_comp);
                });
            } else {
                amrex::ParallelFor(copy_box, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    full_array(i,j,k,n+full_comp) = slice_array(i,j,k,n+slice_comp);
                });
            }
        }
    }
}

void
Fields::ShiftSlices (int lev)
{
    HIPACE_PROFILE("Fields::ShiftSlices()");
    std::swap(m_slices[lev][(int) WhichSlice::Previous1],
              m_slices[lev][(int) WhichSlice::Previous2]);
    std::swap(m_slices[lev][(int) WhichSlice::This],
              m_slices[lev][(int) WhichSlice::Previous1]);
}

amrex::MultiFab
Fields::getF (int lev, int icomp )
{
    amrex::MultiFab F_comp(m_F[lev], amrex::make_alias, icomp, 1);
    return F_comp;
}

void Fields::SolvePoissonExmByAndEypBx (amrex::Geometry const& geom, const MPI_Comm& m_comm_xy,
                                        const int lev)
{
    /* Solves Laplacian(-Psi) =  1/episilon0 * (rho-Jz/c) and
     * calculates Ex-c By, Ey + c Bx from  grad(-Psi)
     */
    HIPACE_PROFILE("Fields::SolveExmByAndEypBx()");

    PhysConst phys_const = get_phys_const();

    // Left-Hand Side for Poisson equation is Psi in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        FieldComps::Psi, 1);

    // calculating the right-hand side 1/episilon0 * (rho-Jz/c)
    amrex::MultiFab::Copy(m_poisson_solver.StagingArea(), getSlices(lev, WhichSlice::This),
                              FieldComps::jz, 0, 1, 0);
    m_poisson_solver.StagingArea().mult(-1./phys_const.c);
    amrex::MultiFab::Add(m_poisson_solver.StagingArea(), getSlices(lev, WhichSlice::This),
                          FieldComps::rho, 0, 1, 0);


    m_poisson_solver.SolvePoissonEquation(lhs);

    /* ---------- Transverse FillBoundary Psi ---------- */
    amrex::ParallelContext::push(m_comm_xy);
    lhs.FillBoundary(geom.periodicity());
    amrex::ParallelContext::pop();

    /* Compute ExmBy and Eypbx from grad(-psi) */
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::x,
        geom.CellSize(Direction::x),
        1.,
        SliceOperatorType::Assign,
        FieldComps::Psi,
        FieldComps::ExmBy);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::y,
        geom.CellSize(Direction::y),
        1.,
        SliceOperatorType::Assign,
        FieldComps::Psi,
        FieldComps::EypBx);
}


void Fields::SolvePoissonEz (amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonEz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        FieldComps::Ez, 1);
    // Right-Hand Side for Poisson equation: compute 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
    // from the slice MF, and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver.StagingArea(),
        Direction::x,
        geom.CellSize(Direction::x),
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Assign,
        FieldComps::jx);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver.StagingArea(),
        Direction::y,
        geom.CellSize(Direction::y),
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Add,
        FieldComps::jy);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(lhs);
}

void Fields::SolvePoissonBx (amrex::MultiFab& Bx_iter, amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(Bx) = mu_0*(- d_y(jz) + d_z(jy) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBx()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute -mu_0*d_y(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver.StagingArea(),
        Direction::y,
        geom.CellSize(Direction::y),
        -phys_const.mu0,
        SliceOperatorType::Assign,
        FieldComps::jz);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver.StagingArea(),
        geom.CellSize(Direction::z),
        phys_const.mu0,
        SliceOperatorType::Add,
        FieldComps::jy, FieldComps::jy);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(Bx_iter);
}

void Fields::SolvePoissonBy (amrex::MultiFab& By_iter, amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(By) = mu_0*(d_x(jz) - d_z(jx) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBy()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute mu_0*d_x(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver.StagingArea(),
        Direction::x,
        geom.CellSize(Direction::x),
        phys_const.mu0,
        SliceOperatorType::Assign,
        FieldComps::jz);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver.StagingArea(),
        geom.CellSize(Direction::z),
        -phys_const.mu0,
        SliceOperatorType::Add,
        FieldComps::jx, FieldComps::jx);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(By_iter);
}

void Fields::SolvePoissonBz (amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(Bz) = mu_0*(d_y(jx) - d_x(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonBz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        FieldComps::Bz, 1);
    // Right-Hand Side for Poisson equation: compute mu_0*(d_y(jx) - d_x(jy))
    // from the slice MF, and store in the staging area of m_poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver.StagingArea(),
        Direction::y,
        geom.CellSize(Direction::y),
        phys_const.mu0,
        SliceOperatorType::Assign,
        FieldComps::jx);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver.StagingArea(),
        Direction::x,
        geom.CellSize(Direction::x),
        -phys_const.mu0,
        SliceOperatorType::Add,
        FieldComps::jy);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver.SolvePoissonEquation(lhs);
}

void Fields::InitialBfieldGuess (const amrex::Real relative_Bfield_error,
                                 const amrex::Real predcorr_B_error_tolerance, const int lev)
{
    /* Sets the initial guess of the B field from the two previous slices
     */
    HIPACE_PROFILE("Fields::InitialBfieldGuess()");

    const amrex::Real mix_factor_init_guess = exp(-0.5 * pow(relative_Bfield_error /
                                              ( 2.5 * predcorr_B_error_tolerance ), 2));

    amrex::MultiFab::LinComb(getSlices(lev, WhichSlice::This), 1+mix_factor_init_guess,
                             getSlices(lev, WhichSlice::Previous1), FieldComps::Bx,
                             -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2),
                             FieldComps::Bx, FieldComps::Bx, 1, 0);

    amrex::MultiFab::LinComb(getSlices(lev, WhichSlice::This), 1+mix_factor_init_guess,
                             getSlices(lev, WhichSlice::Previous1), FieldComps::By,
                             -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2),
                             FieldComps::By, FieldComps::By, 1, 0);

}

void Fields::MixAndShiftBfields (const amrex::MultiFab& B_iter, amrex::MultiFab& B_prev_iter,
                                 const int field_comp, const amrex::Real relative_Bfield_error,
                                 const amrex::Real relative_Bfield_error_prev_iter,
                                 const amrex::Real predcorr_B_mixing_factor, const int lev)
{
    /* Mixes the B field according to B = a*B + (1-a)*( c*B_iter + d*B_prev_iter),
     * with a,c,d mixing coefficients.
     */
    HIPACE_PROFILE("Fields::MixAndShiftBfields()");

    /* Mixing factors to mix the current and previous iteration of the B field */
    amrex::Real weight_B_iter;
    amrex::Real weight_B_prev_iter;
    /* calculating the weight for mixing the current and previous iteration based
     * on their respective errors. Large errors will induce a small weight of and vice-versa  */
    if (relative_Bfield_error != 0.0 || relative_Bfield_error_prev_iter != 0.0)
    {
        weight_B_iter = relative_Bfield_error_prev_iter /
                        ( relative_Bfield_error + relative_Bfield_error_prev_iter );
        weight_B_prev_iter = relative_Bfield_error /
                             ( relative_Bfield_error + relative_Bfield_error_prev_iter );
    }
    else
    {
        weight_B_iter = 0.5;
        weight_B_prev_iter = 0.5;
    }

    /* calculating the mixed temporary B field  B_prev_iter = c*B_iter + d*B_prev_iter.
     * This is temporarily stored in B_prev_iter just to avoid additional memory allocation.
     * B_prev_iter is overwritten at the end of this function */
    amrex::MultiFab::LinComb(B_prev_iter, weight_B_iter, B_iter, 0, weight_B_prev_iter,
                             B_prev_iter, 0, 0, 1, 0);

    /* calculating the mixed B field  B = a*B + (1-a)*B_prev_iter */
    amrex::MultiFab::LinComb(getSlices(lev, WhichSlice::This), 1-predcorr_B_mixing_factor,
                             getSlices(lev, WhichSlice::This), field_comp,
                             predcorr_B_mixing_factor, B_prev_iter, 0, field_comp, 1, 0);

    /* Shifting the B field from the current iteration to the previous iteration */
    amrex::MultiFab::Copy(B_prev_iter, B_iter, 0, 0, 1, 0);

}

amrex::Real Fields::ComputeRelBFieldError (const amrex::MultiFab& Bx,
                                           const amrex::MultiFab& By,
                                           const amrex::MultiFab& Bx_iter,
                                           const amrex::MultiFab& By_iter,
                                           const int Bx_comp, const int By_comp,
                                           const int Bx_iter_comp, const int By_iter_comp,
                                           const amrex::Box& bx, const int lev)
{
    /* calculates the relative B field error between two B fields
     * for both Bx and By simultaneously */
    HIPACE_PROFILE("Fields::ComputeRelBFieldError()");

    /* one temporary array is needed to store the difference of B fields
     * between previous and current iteration */
    amrex::MultiFab temp(getSlices(lev, WhichSlice::This).boxArray(),
                         getSlices(lev, WhichSlice::This).DistributionMap(), 1,
                         getSlices(lev, WhichSlice::This).nGrowVect());
    /* calculating sqrt( |Bx|^2 + |By|^2 ) */
    amrex::Real const norm_B = sqrt(amrex::MultiFab::Dot(Bx, Bx_comp, 1, 0)
                               + amrex::MultiFab::Dot(By, By_comp, 1, 0));

    /* calculating sqrt( |Bx - Bx_prev_iter|^2 + |By - By_prev_iter|^2 ) */
    amrex::MultiFab::Copy(temp, Bx, Bx_comp, 0, 1, 0);
    amrex::MultiFab::Subtract(temp, Bx_iter, Bx_iter_comp, 0, 1, 0);
    amrex::Real norm_Bdiff = amrex::MultiFab::Dot(temp, 0, 1, 0);
    amrex::MultiFab::Copy(temp, By, By_comp, 0, 1, 0);
    amrex::MultiFab::Subtract(temp, By_iter, By_iter_comp, 0, 1, 0);
    norm_Bdiff += amrex::MultiFab::Dot(temp, 0, 1, 0);
    norm_Bdiff = sqrt(norm_Bdiff);

    /* calculating the relative error
     * Warning: this test might be not working in SI units! */
     const amrex::Real relative_Bfield_error = (norm_B/bx.numPts() > 1e-10)
                                                ? norm_Bdiff/norm_B : 0.;

    return relative_Bfield_error;
}
