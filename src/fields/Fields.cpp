#include "Fields.H"
#include "fft_poisson_solver/FFTPoissonSolverPeriodic.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichlet.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"

Fields::Fields (Hipace const* a_hipace)
    : m_slices(a_hipace->maxLevel()+1)
{
    amrex::ParmParse ppf("fields");
    ppf.query("do_dirichlet_poisson", m_do_dirichlet_poisson);
}

void
Fields::AllocData (
    int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
    amrex::Geometry const& geom, const amrex::BoxArray& slice_ba, const amrex::DistributionMapping& slice_dm)
{
    HIPACE_PROFILE("Fields::AllocData()");
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = std::max(1, Hipace::m_depos_order_xy);
    m_slices_nguards = {nguards_xy, nguards_xy, 0};

    // Only xy slices need guard cells, there is no deposition to/gather from the output array F.
    amrex::IntVect nguards_F = amrex::IntVect(0,0,0);

    for (int islice=0; islice<WhichSlice::N; islice++) {
        m_slices[lev][islice].define(
            slice_ba, slice_dm, Comps[islice]["N"], m_slices_nguards,
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[lev][islice].setVal(0.0);
    }

    // The Poisson solver operates on transverse slices only.
    // The constructor takes the BoxArray and the DistributionMap of a slice,
    // so the FFTPlans are built on a slice.
    if (m_do_dirichlet_poisson){
        m_poisson_solver = std::unique_ptr<FFTPoissonSolverDirichlet>(
            new FFTPoissonSolverDirichlet(getSlices(lev, WhichSlice::This).boxArray(),
                                          getSlices(lev, WhichSlice::This).DistributionMap(),
                                          geom));
    } else {
        m_poisson_solver = std::unique_ptr<FFTPoissonSolverPeriodic>(
            new FFTPoissonSolverPeriodic(getSlices(lev, WhichSlice::This).boxArray(),
                                         getSlices(lev, WhichSlice::This).DistributionMap(),
                                         geom));
    }
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

void
Fields::LongitudinalDerivative (const amrex::MultiFab& src1, const amrex::MultiFab& src2,
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
              int ncomp, amrex::FArrayBox& fab, int slice_dir, amrex::Geometry geom)
{
    using namespace amrex::literals;
    HIPACE_PROFILE("Fields::Copy()");
    auto& slice_mf = m_slices[lev][WhichSlice::This]; // copy from/to the current slice
    amrex::Array4<amrex::Real> slice_array; // There is only one Box.
    for (amrex::MFIter mfi(slice_mf); mfi.isValid(); ++mfi) {
        auto& slice_fab = slice_mf[mfi];
        amrex::Box slice_box = slice_fab.box();
        slice_box.setSmall(Direction::z, i_slice);
        slice_box.setBig  (Direction::z, i_slice);
        slice_array = amrex::makeArray4(slice_fab.dataPtr(), slice_box, slice_fab.nComp());
        // slice_array's longitude index is i_slice.
    }

    amrex::Box const& vbx = fab.box();
    if (vbx.smallEnd(Direction::z) <= i_slice and
        vbx.bigEnd  (Direction::z) >= i_slice)
    {
        amrex::Box copy_box = vbx;
        copy_box.setSmall(Direction::z, i_slice);
        copy_box.setBig  (Direction::z, i_slice);

        amrex::Array4<amrex::Real> const& full_array = fab.array();

        const amrex::IntVect ncells_global = geom.Domain().length();
        const bool nx_even = ncells_global[0] % 2 == 0;
        const bool ny_even = ncells_global[1] % 2 == 0;

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
                if        (slice_dir ==-1 /* 3D data */){
                    full_array(i,j,k,n+full_comp) = slice_array(i,j,k,n+slice_comp);
                } else if (slice_dir == 0 /* yz slice */){
                    full_array(i,j,k,n+full_comp) =
                        nx_even ? 0.5_rt * (slice_array(i-1,j,k,n+slice_comp) +
                                            slice_array(i,j,k,n+slice_comp))
                        : slice_array(i,j,k,n+slice_comp);
                } else /* slice_dir == 1, xz slice */{
                    full_array(i,j,k,n+full_comp) =
                        ny_even ? 0.5_rt * ( slice_array(i,j-1,k,n+slice_comp) +
                                             slice_array(i,j,k,n+slice_comp))
                        : slice_array(i,j,k,n+slice_comp);
                }
            });
        }
    }
}

void
Fields::ShiftSlices (int lev)
{
    HIPACE_PROFILE("Fields::ShiftSlices()");
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous2), getSlices(lev, WhichSlice::Previous1),
        Comps[WhichSlice::Previous1]["Bx"], Comps[WhichSlice::Previous2]["Bx"],
        2, m_slices_nguards);
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous1), getSlices(lev, WhichSlice::This),
        Comps[WhichSlice::This]["Bx"], Comps[WhichSlice::Previous1]["Bx"],
        2, m_slices_nguards);
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous1), getSlices(lev, WhichSlice::This),
        Comps[WhichSlice::This]["jx"], Comps[WhichSlice::Previous1]["jx"],
        4, m_slices_nguards);
}

void
Fields::AddRhoIons (const int lev, bool inverse)
{
    HIPACE_PROFILE("Fields::AddRhoIons()");
    if (!inverse){
        amrex::MultiFab::Add(getSlices(lev, WhichSlice::This), getSlices(lev, WhichSlice::RhoIons),
                             Comps[WhichSlice::RhoIons]["rho"], Comps[WhichSlice::This]["rho"], 1, 0);
    } else {
        amrex::MultiFab::Subtract(getSlices(lev, WhichSlice::This), getSlices(lev, WhichSlice::RhoIons),
                                  Comps[WhichSlice::RhoIons]["rho"], Comps[WhichSlice::This]["rho"], 1, 0);
    }
}

void
Fields::AddBeamCurrents (const int lev, const int which_slice)
{
    HIPACE_PROFILE("Fields::AddBeamCurrents()");
    amrex::MultiFab& S = getSlices(lev, which_slice);
    // we add the beam currents to the full currents, as mostly the full currents are needed
    amrex::MultiFab::Add(S, S, Comps[which_slice]["jx_beam"], Comps[which_slice]["jx"], 1,
                         {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});
    amrex::MultiFab::Add(S, S, Comps[which_slice]["jy_beam"], Comps[which_slice]["jy"], 1,
                         {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});
    if (which_slice == WhichSlice::This) {
        amrex::MultiFab::Add(S, S, Comps[which_slice]["jz_beam"], Comps[which_slice]["jz"], 1,
                             {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});
    }
}

void
Fields::SolvePoissonExmByAndEypBx (amrex::Geometry const& geom, const MPI_Comm& m_comm_xy,
                                   const int lev)
{
    /* Solves Laplacian(Psi) =  1/episilon0 * -(rho-Jz/c) and
     * calculates Ex-c By, Ey + c Bx from  grad(-Psi)
     */
    HIPACE_PROFILE("Fields::SolveExmByAndEypBx()");

    PhysConst phys_const = get_phys_const();

    // Left-Hand Side for Poisson equation is Psi in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Psi"], 1);

    // calculating the right-hand side 1/episilon0 * -(rho-Jz/c)
    amrex::MultiFab::Copy(m_poisson_solver->StagingArea(), getSlices(lev, WhichSlice::This),
                              Comps[WhichSlice::This]["jz"], 0, 1, 0);
    m_poisson_solver->StagingArea().mult(-1./phys_const.c);
    amrex::MultiFab::Add(m_poisson_solver->StagingArea(), getSlices(lev, WhichSlice::This),
                          Comps[WhichSlice::This]["rho"], 0, 1, 0);
    m_poisson_solver->StagingArea().mult(-1./phys_const.ep0);

    m_poisson_solver->SolvePoissonEquation(lhs);

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
        -1.,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["Psi"],
        Comps[WhichSlice::This]["ExmBy"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::y,
        geom.CellSize(Direction::y),
        -1.,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["Psi"],
        Comps[WhichSlice::This]["EypBx"]);
}


void
Fields::SolvePoissonEz (amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonEz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Ez"], 1);
    // Right-Hand Side for Poisson equation: compute 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
    // from the slice MF, and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver->StagingArea(),
        Direction::x,
        geom.CellSize(Direction::x),
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jx"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver->StagingArea(),
        Direction::y,
        geom.CellSize(Direction::y),
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Add,
        Comps[WhichSlice::This]["jy"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver->SolvePoissonEquation(lhs);
}

void
Fields::SolvePoissonBx (amrex::MultiFab& Bx_iter, amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(Bx) = mu_0*(- d_y(jz) + d_z(jy) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBx()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute -mu_0*d_y(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver->StagingArea(),
        Direction::y,
        geom.CellSize(Direction::y),
        -phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jz"]);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver->StagingArea(),
        geom.CellSize(Direction::z),
        phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::Previous1]["jy"],
        Comps[WhichSlice::Next]["jy"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver->SolvePoissonEquation(Bx_iter);
}

void
Fields::SolvePoissonBy (amrex::MultiFab& By_iter, amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(By) = mu_0*(d_x(jz) - d_z(jx) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBy()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute mu_0*d_x(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver->StagingArea(),
        Direction::x,
        geom.CellSize(Direction::x),
        phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jz"]);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver->StagingArea(),
        geom.CellSize(Direction::z),
        -phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::Previous1]["jx"],
        Comps[WhichSlice::Next]["jx"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver->SolvePoissonEquation(By_iter);
}

void
Fields::SolvePoissonBz (amrex::Geometry const& geom, const int lev)
{
    /* Solves Laplacian(Bz) = mu_0*(d_y(jx) - d_x(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonBz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Bz"], 1);
    // Right-Hand Side for Poisson equation: compute mu_0*(d_y(jx) - d_x(jy))
    // from the slice MF, and store in the staging area of m_poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver->StagingArea(),
        Direction::y,
        geom.CellSize(Direction::y),
        phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jx"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver->StagingArea(),
        Direction::x,
        geom.CellSize(Direction::x),
        -phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::This]["jy"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver->SolvePoissonEquation(lhs);
}

void
Fields::InitialBfieldGuess (const amrex::Real relative_Bfield_error,
                            const amrex::Real predcorr_B_error_tolerance, const int lev)
{
    /* Sets the initial guess of the B field from the two previous slices
     */
    HIPACE_PROFILE("Fields::InitialBfieldGuess()");

    const amrex::Real mix_factor_init_guess = exp(-0.5 * pow(relative_Bfield_error /
                                              ( 2.5 * predcorr_B_error_tolerance ), 2));

    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1+mix_factor_init_guess, getSlices(lev, WhichSlice::Previous1), Comps[WhichSlice::Previous1]["Bx"],
        -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2), Comps[WhichSlice::Previous2]["Bx"],
        Comps[WhichSlice::This]["Bx"], 1, 0);

    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1+mix_factor_init_guess, getSlices(lev, WhichSlice::Previous1), Comps[WhichSlice::Previous1]["By"],
        -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2), Comps[WhichSlice::Previous2]["By"],
        Comps[WhichSlice::This]["By"], 1, 0);
}

void
Fields::MixAndShiftBfields (const amrex::MultiFab& B_iter, amrex::MultiFab& B_prev_iter,
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
    amrex::MultiFab::LinComb(
        B_prev_iter,
        weight_B_iter, B_iter, 0,
        weight_B_prev_iter, B_prev_iter, 0,
        0, 1, 0);

    /* calculating the mixed B field  B = a*B + (1-a)*B_prev_iter */
    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1-predcorr_B_mixing_factor, getSlices(lev, WhichSlice::This), field_comp,
        predcorr_B_mixing_factor, B_prev_iter, 0,
        field_comp, 1, 0);

    /* Shifting the B field from the current iteration to the previous iteration */
    amrex::MultiFab::Copy(B_prev_iter, B_iter, 0, 0, 1, 0);

}

amrex::Real
Fields::ComputeRelBFieldError (
    const amrex::MultiFab& Bx, const amrex::MultiFab& By, const amrex::MultiFab& Bx_iter,
    const amrex::MultiFab& By_iter, const int Bx_comp, const int By_comp, const int Bx_iter_comp,
    const int By_iter_comp, const amrex::Geometry& geom)
{
    // calculates the relative B field error between two B fields
    // for both Bx and By simultaneously
    HIPACE_PROFILE("Fields::ComputeRelBFieldError()");

    amrex::Real norm_Bdiff = 0;
    amrex::Gpu::DeviceScalar<amrex::Real> gpu_norm_Bdiff(norm_Bdiff);
    amrex::Real* p_norm_Bdiff = gpu_norm_Bdiff.dataPtr();

    amrex::Real norm_B = 0;
    amrex::Gpu::DeviceScalar<amrex::Real> gpu_norm_B(norm_B);
    amrex::Real* p_norm_B = gpu_norm_B.dataPtr();

    for ( amrex::MFIter mfi(Bx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & Bx_array = Bx.array(mfi);
        amrex::Array4<amrex::Real const> const & Bx_iter_array = Bx_iter.array(mfi);
        amrex::Array4<amrex::Real const> const & By_array = By.array(mfi);
        amrex::Array4<amrex::Real const> const & By_iter_array = By_iter.array(mfi);

        amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept
        {
            amrex::Gpu::deviceReduceSum(p_norm_B, std::sqrt(
                                        Bx_array(i, j, k, Bx_comp) * Bx_array(i, j, k, Bx_comp) +
                                        By_array(i, j, k, By_comp) * By_array(i, j, k, By_comp)),
                                        handler);
            amrex::Gpu::deviceReduceSum(p_norm_Bdiff, std::sqrt(
                            ( Bx_array(i, j, k, Bx_comp) - Bx_iter_array(i, j, k, Bx_iter_comp) ) *
                            ( Bx_array(i, j, k, Bx_comp) - Bx_iter_array(i, j, k, Bx_iter_comp) ) +
                            ( By_array(i, j, k, By_comp) - By_iter_array(i, j, k, By_iter_comp) ) *
                            ( By_array(i, j, k, By_comp) - By_iter_array(i, j, k, By_iter_comp) )),
                            handler);
        }
        );
    }
    // no cudaDeviceSynchronize required here, as there is one in the MFIter destructor called above.
    norm_Bdiff = gpu_norm_Bdiff.dataValue();
    norm_B = gpu_norm_B.dataValue();

    const int numPts_transverse = geom.Domain().length(0) * geom.Domain().length(1);

    // calculating the relative error
    // Warning: this test might be not working in SI units!
    const amrex::Real relative_Bfield_error = (norm_B/numPts_transverse > 1e-10)
                                               ? norm_Bdiff/norm_B : 0.;

    return relative_Bfield_error;
}
