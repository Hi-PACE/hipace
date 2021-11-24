#include "Fields.H"
#include "fft_poisson_solver/FFTPoissonSolverPeriodic.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichlet.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "particles/ShapeFactors.H"

amrex::IntVect Fields::m_slices_nguards = {-1, -1, -1};
amrex::IntVect Fields::m_valid_nguards = {-1, -1, -1};

amrex::Vector<amrex::Box> Fields::m_box_problem{};
amrex::Vector<amrex::Box> Fields::m_box_source{};
amrex::Vector<amrex::Box> Fields::m_box_valid{};
amrex::Vector<amrex::Box> Fields::m_box_all{};

Fields::Fields (Hipace const* a_hipace)
    : m_slices(a_hipace->maxLevel()+1)
{
    const int max_lev = a_hipace->maxLevel()+1;
    m_box_problem.resize(max_lev);
    m_box_source.resize(max_lev);
    m_box_valid.resize(max_lev);
    m_box_all.resize(max_lev);
    amrex::ParmParse ppf("fields");
    queryWithParser(ppf, "do_dirichlet_poisson", m_do_dirichlet_poisson);
}

void
Fields::AllocData (
    int lev, amrex::Vector<amrex::Geometry> const& geom, const amrex::BoxArray& slice_ba,
    const amrex::DistributionMapping& slice_dm, int bin_size)
{
    HIPACE_PROFILE("Fields::AllocData()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(slice_ba.size() == 1,
        "Parallel field solvers not supported yet");

    // Need 1 extra guard cell transversally for transverse derivative of Psi
    int nguards_xy = Hipace::m_depos_order_xy;
    m_valid_nguards = {nguards_xy, nguards_xy, 0};
    m_slices_nguards = {nguards_xy + 1, nguards_xy + 1, 0};

    // box where the problem is defined
    m_box_problem[lev] = slice_ba[0];

    // source terms of derivatives can be only in here
    m_box_source[lev] = m_box_problem[lev];
    m_box_source[lev].grow({-1, -1, 0});

    // contains valid fileds that can be used for particles, also the box of the Poisson solver
    m_box_valid[lev] = m_box_problem[lev];
    m_box_valid[lev].grow(m_valid_nguards);

    // valid box for Psi
    m_box_all[lev] = m_box_problem[lev];
    m_box_all[lev].grow(m_slices_nguards);

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
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichlet>(
            new FFTPoissonSolverDirichlet(getSlices(lev, WhichSlice::This).boxArray(),
                                          getSlices(lev, WhichSlice::This).DistributionMap(),
                                          geom[lev])) );
    } else {
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverPeriodic>(
            new FFTPoissonSolverPeriodic(getSlices(lev, WhichSlice::This).boxArray(),
                                         getSlices(lev, WhichSlice::This).DistributionMap(),
                                         geom[lev]))  );
    }
    int num_threads = 1;
#ifdef AMREX_USE_OMP
#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
#endif
    if (Hipace::m_do_tiling) {
        const amrex::Box dom_box = slice_ba[0];
        const amrex::IntVect ncell = dom_box.bigEnd() - dom_box.smallEnd() + 1;
        AMREX_ALWAYS_ASSERT(ncell[0] % bin_size == 0 && ncell[1] % bin_size == 0);

        m_tmp_densities.resize(num_threads);
        for (int i=0; i<num_threads; i++){
            amrex::Box bx = {{0, 0, 0}, {bin_size-1, bin_size-1, 0}};
            bx.grow(m_slices_nguards);
            // jx jy jz rho jxx jxy jyy
            m_tmp_densities[i].resize(bx, 7);
        }
    }
}


struct derivative_x_GPU {
    amrex::Array4<amrex::Real const> array;
    amrex::Real dx_inv;

    AMREX_GPU_DEVICE amrex::Real operator ()(int i, int j, int k) const {
        return (array(i+1,j,k) - array(i-1,j,k)) * dx_inv;
    }
};

struct derivative_y_GPU {
    amrex::Array4<amrex::Real const> array;
    amrex::Real dy_inv;

    AMREX_GPU_DEVICE amrex::Real operator ()(int i, int j, int k) const {
        return (array(i,j+1,k) - array(i,j-1,k)) * dy_inv;
    }
};

struct derivative_z_GPU {
    amrex::Array4<amrex::Real const> array1;
    amrex::Array4<amrex::Real const> array2;
    amrex::Real dz_inv;

    AMREX_GPU_DEVICE amrex::Real operator ()(int i, int j, int k) const {
        return (array1(i,j,k) - array2(i,j,k)) * dz_inv;
    }
};



struct derivative_x {
    FieldView f_view;
    amrex::Real dx;

    derivative_x_GPU array (amrex::MFIter& mfi) const {
        return derivative_x_GPU{f_view.array(mfi), 1/(2*dx)};
    }
};

struct derivative_y {
    FieldView f_view;
    amrex::Real dy;

    derivative_y_GPU array (amrex::MFIter& mfi) const {
        return derivative_y_GPU{f_view.array(mfi), 1/(2*dy)};
    }
};

struct derivative_z {
    FieldView f_view1;
    FieldView f_view2;
    amrex::Real dz;

    derivative_z_GPU array (amrex::MFIter& mfi) const {
        return derivative_z_GPU{f_view1.array(mfi), f_view2.array(mfi), 1/(2*dz)};
    }
};

template<class FVA, class FVB>
void
FieldOperation (const amrex::Box op_box, FieldView dst,
                const amrex::Real factor_a, const FVA src_a,
                const amrex::Real factor_b, const FVB src_b,
                const amrex::Box valid_box)
{
    HIPACE_PROFILE("Fields::FieldOperation()");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(dst.m_mfab, amrex::TilingIfNotGPU());
          mfi.isValid(); ++mfi ){
        const auto& dst_array = dst.array(mfi);
        const auto src_a_array = src_a.array(mfi);
        const auto src_b_array = src_b.array(mfi);
        const amrex::Box bx = mfi.tilebox() & op_box;
        const int i_lo = valid_box.smallEnd(0);
        const int i_hi = valid_box.bigEnd(0);
        const int j_lo = valid_box.smallEnd(1);
        const int j_hi = valid_box.bigEnd(1);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                dst_array(i,j,k) = (factor_a*src_a_array(i,j,k) + factor_b*src_b_array(i,j,k))
                                  *(i_lo<=i && i<=i_hi && j_lo<=j && j<=j_hi);
            });
    }
}

template<class FV>
void
UnaryFieldOperation (const amrex::Box op_box, FieldView dst,
                     const amrex::Real factor, const FV src)
{
    HIPACE_PROFILE("Fields::FieldOperation()");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(dst.m_mfab, amrex::TilingIfNotGPU());
          mfi.isValid(); ++mfi ){
        const auto& dst_array = dst.array(mfi);
        const auto src_array = src.array(mfi);
        const amrex::Box bx = mfi.tilebox() & op_box;

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                dst_array(i,j,k) = factor*src_array(i,j,k);
            });
    }
}

/*
void
Fields::CopyToStagingArea (const amrex::MultiFab& src, const SliceOperatorType slice_operator,
                           const int scomp, const int lev)
{
    HIPACE_PROFILE("Fields::CopyToStagingArea()");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(m_poisson_solver[lev]->StagingArea(), amrex::TilingIfNotGPU());
          mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = m_poisson_solver[lev]
                                                       ->StagingArea().array(mfi);
        amrex::Box prob_box = m_box_source[lev];

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (slice_operator==SliceOperatorType::Assign) {
                    dst_array(i,j,k,0) = src_array(i,j,k,scomp) * prob_box.contains(i,j,k);
                }
                else
                {
                    dst_array(i,j,k,0) += src_array(i,j,k,scomp) * prob_box.contains(i,j,k);
                }
            }
            );
    }
}

void
Fields::TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                              const amrex::Real dx, const int lev, const amrex::Real mult_coeff,
                              const SliceOperatorType slice_operator, const int scomp,
                              const int dcomp, const bool use_offset)
{
    HIPACE_PROFILE("Fields::TransverseDerivative()");
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(src.size() == 1, "Slice MFs must be defined on one box only");
    AMREX_ALWAYS_ASSERT((direction == Direction::x) || (direction == Direction::y));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (direction == Direction::x){

                    if (slice_operator==SliceOperatorType::Assign)
                    {
                        dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dx) *
                                                 (src_array(i+1, j, k, scomp)
                                                  - src_array(i-1, j, k, scomp));
                    }
                    else
                    {
                        dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dx) *
                                                  (src_array(i+1, j, k, scomp)
                                                   - src_array(i-1, j, k, scomp));
                    }
                } else  {

                    if (slice_operator==SliceOperatorType::Assign)
                    {
                        dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dx) *
                                                 (src_array(i, j+1, k, scomp)
                                                  - src_array(i, j-1, k, scomp));
                    }

                    {
                        dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dx) *
                                                  (src_array(i, j+1, k, scomp)
                                                   - src_array(i, j-1, k, scomp));
                    }
                }
            }
            );
    }
}

void
Fields::LongitudinalDerivative (const amrex::MultiFab& src1, const amrex::MultiFab& src2,
                                amrex::MultiFab& dst, const amrex::Real dz, const int lev,
                                const amrex::Real mult_coeff,
                                const SliceOperatorType slice_operator, const int s1comp,
                                const int s2comp, const int dcomp)
{
    HIPACE_PROFILE("Fields::LongitudinalDerivative()");
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(src1.size() == 1, "Slice MFs must be defined on one box only");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
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
                                             (src1_array(i, j, k, s1comp)
                                              - src2_array(i, j, k, s2comp));
                }
                else
                {
                    dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dz) *
                                              (src1_array(i, j, k, s1comp)
                                               - src2_array(i, j, k, s2comp));
                }

            }
            );
    }
}
*/

void
Fields::Copy (const int lev, const int i_slice, const int slice_comp, const int full_comp,
              const amrex::Gpu::DeviceVector<int>& diag_comps_vect,
              const int ncomp, amrex::FArrayBox& fab, const int slice_dir,
              const amrex::IntVect diag_coarsen, const amrex::Geometry geom)
{
    using namespace amrex::literals;
    HIPACE_PROFILE("Fields::Copy()");
    auto& slice_mf = m_slices[lev][WhichSlice::This]; // copy from the current slice
    amrex::Array4<amrex::Real> slice_array; // There is only one Box.
    for (amrex::MFIter mfi(slice_mf); mfi.isValid(); ++mfi) {
        auto& slice_fab = slice_mf[mfi];
        amrex::Box slice_box = slice_fab.box();
        slice_box.setSmall(Direction::z, i_slice);
        slice_box.setBig  (Direction::z, i_slice);
        slice_array = amrex::makeArray4(slice_fab.dataPtr(), slice_box, slice_fab.nComp());
        // slice_array's longitude index is i_slice.
    }

    const int full_array_z = i_slice / diag_coarsen[2];
    const amrex::IntVect ncells_global = geom.Domain().length();

    amrex::Box const& vbx = fab.box();
    if (vbx.smallEnd(Direction::z) <= full_array_z and
        vbx.bigEnd  (Direction::z) >= full_array_z and
        ( i_slice % diag_coarsen[2] == diag_coarsen[2]/2 or
        ( i_slice == ncells_global[2] - 1 and
        ( ncells_global[2] - 1 ) % diag_coarsen[2] < diag_coarsen[2]/2 )))
    {
        amrex::Box copy_box = vbx;
        copy_box.setSmall(Direction::z, full_array_z);
        copy_box.setBig  (Direction::z, full_array_z);

        amrex::Array4<amrex::Real> const& full_array = fab.array();

        const int even_slice_x = ncells_global[0] % 2 == 0 and slice_dir == 0;
        const int even_slice_y = ncells_global[1] % 2 == 0 and slice_dir == 1;

        const int coarse_x = diag_coarsen[0];
        const int coarse_y = diag_coarsen[1];

        const int ncells_x = ncells_global[0];
        const int ncells_y = ncells_global[1];

        const int *diag_comps = diag_comps_vect.data();

        amrex::ParallelFor(copy_box, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int m = n[diag_comps];

            // coarsening in slice direction is always 1
            const int i_c_start = amrex::min(i*coarse_x +(coarse_x-1)/2 -even_slice_x, ncells_x-1);
            const int i_c_stop  = amrex::min(i*coarse_x +coarse_x/2+1, ncells_x);
            const int j_c_start = amrex::min(j*coarse_y +(coarse_y-1)/2 -even_slice_y, ncells_y-1);
            const int j_c_stop  = amrex::min(j*coarse_y +coarse_y/2+1, ncells_y);

            amrex::Real field_value = 0._rt;
            int n_values = 0;

            for (int j_c = j_c_start; j_c != j_c_stop; ++j_c) {
                for (int i_c = i_c_start; i_c != i_c_stop; ++i_c) {
                    field_value += slice_array(i_c, j_c, i_slice, m+slice_comp);
                    ++n_values;
                }
            }

            full_array(i,j,k,n+full_comp) = field_value / amrex::max(n_values,1);
        });
    }
}

void
Fields::ShiftSlices (int nlev, int islice, amrex::Geometry geom, amrex::Real patch_lo,
                     amrex::Real patch_hi)
{
    HIPACE_PROFILE("Fields::ShiftSlices()");

    for (int lev = 0; lev < nlev; ++lev) {

    // skip all slices which are not existing on level 1
    if (lev == 1) {
        // use geometry of coarse grid to determine whether slice is to be solved
        const amrex::Real* problo = geom.ProbLo();
        const amrex::Real* dx = geom.CellSize();
        const amrex::Real pos = (islice+0.5)*dx[2]+problo[2];
        if (pos < patch_lo || pos > patch_hi) continue;
    }

    // shift Bx, By
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous2), getSlices(lev, WhichSlice::Previous1),
        Comps[WhichSlice::Previous1]["Bx"], Comps[WhichSlice::Previous2]["Bx"],
        2, m_slices_nguards);
    // shift Ez, Bx, By, Bz, jx, jx_beam, jy, jy_beam
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous1), getSlices(lev, WhichSlice::This),
        Comps[WhichSlice::This]["Ez"], Comps[WhichSlice::Previous1]["Ez"],
        8, m_slices_nguards);
    // shift rho, Psi
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous1), getSlices(lev, WhichSlice::This),
        Comps[WhichSlice::This]["rho"], Comps[WhichSlice::Previous1]["rho"],
        2, m_slices_nguards);
    }
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
        m_slices_nguards);
    amrex::MultiFab::Add(S, S, Comps[which_slice]["jy_beam"], Comps[which_slice]["jy"], 1,
        m_slices_nguards);
    if (which_slice == WhichSlice::This) {
        amrex::MultiFab::Add(S, S, Comps[which_slice]["jz_beam"], Comps[which_slice]["jz"], 1,
            m_slices_nguards);
    }
}

void
Fields::InterpolateBoundaries (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                               std::string component, const int islice)
{
    // To solve a Poisson equation with non-zero Dirichlet boundary conditions, the source term
    // must be corrected at the outmost grid points in x by -field_value_at_guard_cell / dx^2 and
    // in y by -field_value_at_guard_cell / dy^2, where dx and dy are those of the fine grid
    // This follows Van Loan, C. (1992). Computational frameworks for the fast Fourier transform.
    // Page 254 ff.
    // The interpolation is done in second order transversely and linearly in longitudinal direction

    HIPACE_PROFILE("Fields::InterpolateBoundaries()");
    if (lev == 0) return; // only interpolate boundaries to lev 1
    using namespace amrex::literals;
    const auto plo = geom[lev].ProbLoArray();
    const auto dx = geom[lev].CellSizeArray();
    const auto plo_coarse = geom[lev-1].ProbLoArray();
    const auto dx_coarse = geom[lev-1].CellSizeArray();
constexpr int interp_order = 2;

    // get relative position of fine grid slice between coarse grids for longitudinal lin. interpol.
     const amrex::Real z = plo_coarse[2] + (islice+0.5_rt)*dx[2];
     const int idz_coarse = (z-plo_coarse[2])/dx_coarse[2];
     const amrex::Real rel_z = (z - (plo_coarse[2] + (idz_coarse)*dx_coarse[2])) / dx_coarse[2];

    // get level 0 for interpolation to source term of level 1
    amrex::MultiFab lhs_coarse(getSlices(lev-1, WhichSlice::This), amrex::make_alias,
                               Comps[WhichSlice::This][component], 1);
    amrex::MultiFab lhs_coarse_prev(getSlices(lev-1, WhichSlice::Previous1), amrex::make_alias,
                               Comps[WhichSlice::Previous1][component], 1);
    amrex::FArrayBox& lhs_fab = lhs_coarse[0];
    amrex::Box lhs_bx = lhs_fab.box();
    lhs_bx.grow({-m_slices_nguards[0], -m_slices_nguards[1], 0});
    // low end of the coarse grid excluding guard cells
    const amrex::IntVect lo_coarse = lhs_bx.smallEnd();

    // get offset of level 1 w.r.t. the staging area
    amrex::MultiFab lhs_fine(getSlices(lev, WhichSlice::This), amrex::make_alias,
                              Comps[WhichSlice::This][component], 1);
    amrex::FArrayBox& lhs_fine_fab = lhs_fine[0];
    amrex::Box lhs_fine_bx = lhs_fine_fab.box();
    lhs_fine_bx.grow({-m_slices_nguards[0], -m_slices_nguards[1], 0});
    // low end of the fine grid excluding guard cells, in units of fine cells.
    const amrex::IntVect lo = lhs_fine_bx.smallEnd();

    for (amrex::MFIter mfi( m_poisson_solver[lev]->StagingArea(),false); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.tilebox();
        // Get the big end of the Box
        const amrex::IntVect& big = bx.bigEnd();
        // highest valid index (not counting guard cells) of the staging area in x and y
        const int nx_fine_high = big[0];
        const int ny_fine_high = big[1];
        amrex::Array4<amrex::Real>  data_array = m_poisson_solver[lev]->StagingArea().array(mfi);
        amrex::Array4<amrex::Real>  arr_coarse = lhs_coarse.array(mfi);
        amrex::Array4<amrex::Real>  arr_coarse_prev = lhs_coarse_prev.array(mfi);

        // Loop over the valid indices on the fine grid and interpolate the value of the coarse grid
        // at the location of the guard cell on the fine grid to the first/last valid grid point on
        // the fine grid
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j , int k) noexcept
            {
                if (i==0 || i== nx_fine_high || j==0 || j == ny_fine_high) {
                    // Compute coordinate on fine grid
                    amrex::Real x, y;

                    // handling of the left and right boundary of the staging area
                    if ((i==0) || (i==nx_fine_high)) {
                        if (i==0) {
                            // position of guard cell left of first valid grid point
                            x = plo[0] + (i+lo[0]-0.5_rt)*dx[0];
                        } else if (i== nx_fine_high) {
                            // position of guard cell right of last valid grid point
                            x = plo[0] + (i+lo[0]+1.5_rt)*dx[0];
                        }
                        y = plo[1] + (j+lo[1]+0.5_rt)*dx[1];

                        // --- Compute shape factors
                        // x direction
                        // j_cell leftmost cell in x that the particle touches.
                        // sx_cell shape factor along x
                        const amrex::Real xmid = (x - plo_coarse[0])/dx_coarse[0];
                        amrex::Real sx_cell[interp_order + 1];
                        const int j_cell = compute_shape_factor<interp_order>(sx_cell, xmid-0.5_rt);

                        // y direction
                        const amrex::Real ymid = (y - plo_coarse[1])/dx_coarse[1];
                        amrex::Real sy_cell[interp_order + 1];
                        const int k_cell = compute_shape_factor<interp_order>(sy_cell, ymid-0.5_rt);

                        amrex::Real boundary_value = 0.0_rt;
                        // add interpolated contribution to boundary value
                        for (int iy=0; iy<=interp_order; iy++){
                            for (int ix=0; ix<=interp_order; ix++){
                                boundary_value += sx_cell[ix]*sy_cell[iy]*
                                  ((1.0_rt-rel_z)*arr_coarse(lo_coarse[0]+j_cell+ix,
                                                             lo_coarse[1]+k_cell+iy, lo_coarse[2])
                                     + rel_z*arr_coarse_prev(lo_coarse[0]+j_cell+ix,
                                                             lo_coarse[1]+k_cell+iy, lo_coarse[2]));
                            }
                        }

                        // adjusting source term to get non-zero Dirichlet boundary condition
                        data_array(i,j,k) -= boundary_value/(dx[0]*dx[0]);
                    }

                    // handling of the bottom and top boundary of the staging area
                    if ((j==0) || (j==ny_fine_high)) {
                        if (j==0) {
                            // position of guard cell below of first valid grid point
                            y = plo[1] + (j+lo[1]-0.5_rt)*dx[1];
                        } else if (j== ny_fine_high) {
                            // position of guard cell above of last valid grid point
                            y = plo[1] + (j+lo[1]+1.5_rt)*dx[1];
                        }
                        x = plo[0] + (i+lo[0]+0.5_rt)*dx[0];

                        // --- Compute shape factors
                        // x direction
                        // j_cell leftmost cell in x that the particle touches.
                        // sx_cell shape factor along x
                        const amrex::Real xmid = (x - plo_coarse[0])/dx_coarse[0];
                        amrex::Real sx_cell[interp_order + 1];
                        const int j_cell = compute_shape_factor<interp_order>(sx_cell, xmid-0.5_rt);

                        // y direction
                        const amrex::Real ymid = (y - plo_coarse[1])/dx_coarse[1];
                        amrex::Real sy_cell[interp_order + 1];
                        const int k_cell = compute_shape_factor<interp_order>(sy_cell, ymid-0.5_rt);

                        amrex::Real boundary_value = 0.0_rt;
                        // add interpolated contribution to boundary value
                        for (int iy=0; iy<=interp_order; iy++){
                            for (int ix=0; ix<=interp_order; ix++){
                                boundary_value += sx_cell[ix]*sy_cell[iy]*
                                  ((1.0_rt-rel_z)*arr_coarse(lo_coarse[0]+j_cell+ix,
                                                             lo_coarse[1]+k_cell+iy, lo_coarse[2])
                                     + rel_z*arr_coarse_prev(lo_coarse[0]+j_cell+ix,
                                                             lo_coarse[1]+k_cell+iy, lo_coarse[2]));
                            }
                        }

                        // adjusting source term to get non-zero Dirichlet boundary condition
                        data_array(i,j,k) -= boundary_value/(dx[1]*dx[1]);
                    }
                }
            });
    }
}

void
Fields::InterpolateFromLev0toLev1 (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                                   std::string component, const int islice)
{
    // This function interpolates values from the coarse to the fine grid with second order.
    // This is required for rho to fix the incomplete deposition close to the boundary and for Psi
    // to fill the guard cell, which is needed for the transverse derivative
    // The interpolation is done in second order transversely and linearly in longitudinal direction

    HIPACE_PROFILE("Fields::InterpolateFromLev0toLev1()");
    if (lev == 0) return; // only interpolate boundaries to lev 1
    using namespace amrex::literals;
    const auto plo = geom[lev].ProbLoArray();
    const auto dx = geom[lev].CellSizeArray();
    const auto plo_coarse = geom[lev-1].ProbLoArray();
    const auto dx_coarse = geom[lev-1].CellSizeArray();
    constexpr int interp_order = 2;

    // get relative position of fine grid slice between coarse grids for longitudinal lin. interpol.
     const amrex::Real z = plo_coarse[2] + (islice+0.5_rt)*dx[2];
     const int idz_coarse = (z-plo_coarse[2])/dx_coarse[2];
     const amrex::Real rel_z = (z - (plo_coarse[2] + (idz_coarse)*dx_coarse[2])) / dx_coarse[2];

    // get level 0 array
    amrex::MultiFab lhs_coarse(getSlices(lev-1, WhichSlice::This), amrex::make_alias,
                               Comps[WhichSlice::This][component], 1);
    amrex::MultiFab lhs_coarse_prev(getSlices(lev-1, WhichSlice::Previous1), amrex::make_alias,
                              Comps[WhichSlice::Previous1][component], 1);
    amrex::FArrayBox& lhs_fab = lhs_coarse[0];
    amrex::Box lhs_bx = lhs_fab.box();
    // lhs_bx should only have valid cells
    lhs_bx.grow({-m_slices_nguards[0], -m_slices_nguards[1], 0});
    // low end of the coarse grid excluding guard cells, in units of coarse cells.
    const amrex::IntVect lo_coarse = lhs_bx.smallEnd();

    // get level 1 array
    amrex::MultiFab lhs_fine(getSlices(lev, WhichSlice::This), amrex::make_alias,
                              Comps[WhichSlice::This][component], 1);

    for (amrex::MFIter mfi( lhs_fine,false); mfi.isValid(); ++mfi)
    {
        amrex::Box bx = mfi.tilebox();
        // psi needs the guard cells, as these are the cells we need to fill
        if (component == "Psi") bx.grow(m_slices_nguards);
        // Get the small end of the Box
        const amrex::IntVect& small = bx.smallEnd();
        // the interpolation of rho at the low end starts at the lowest valid cell,
        // for Psi at the guard cell below the first valid cell
        const int nx_fine_low = (component == "rho") ? small[0] : small[0]+m_slices_nguards[0]-1;
        const int ny_fine_low = (component == "rho") ? small[1] : small[1]+m_slices_nguards[1]-1;
        // Get the big end of the Box
        const amrex::IntVect& big = bx.bigEnd();
        // the interpolation of rho at the high end starts at the highest valid cell,
        // for Psi at the guard cell above the last valid cell
        const int nx_fine_high = (component == "rho") ? big[0] : big[0]-m_slices_nguards[0]+1;
        const int ny_fine_high = (component == "rho") ? big[1] : big[1]-m_slices_nguards[0]+1;
        // rho needs to be interpolated for the number of guard cells, Psi just for one guard cell
        const int x_range = (component == "rho") ? m_slices_nguards[0] : 1;
        const int y_range = (component == "rho") ? m_slices_nguards[1] : 1;

        amrex::Array4<amrex::Real>  data_array = lhs_fine.array(mfi);
        amrex::Array4<amrex::Real>  arr_coarse = lhs_coarse.array(mfi);
        amrex::Array4<amrex::Real>  arr_coarse_prev = lhs_coarse_prev.array(mfi);

        // Loop over the valid indices on the fine grid and interpolate the value of the coarse grid
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j , int k) noexcept
            {
                if ((i >= nx_fine_low  && i < nx_fine_low  + x_range) ||
                    (i <= nx_fine_high && i > nx_fine_high - x_range) ||
                    (j >= ny_fine_low  && j < ny_fine_low  + y_range) ||
                    (j <= ny_fine_high && j > ny_fine_high - y_range) ) {

                    const amrex::Real x = plo[0] + (i+0.5_rt)*dx[0];
                    const amrex::Real y = plo[1] + (j+0.5_rt)*dx[1];

                    // --- Compute shape factors
                    // x direction
                    // j_cell leftmost cell in x that the particle touches.
                    // sx_cell shape factor along x
                    const amrex::Real xmid = (x - plo_coarse[0])/dx_coarse[0];
                    amrex::Real sx_cell[interp_order + 1];
                    const int j_cell = compute_shape_factor<interp_order>(sx_cell, xmid-0.5_rt);

                    // y direction
                    const amrex::Real ymid = (y - plo_coarse[1])/dx_coarse[1];
                    amrex::Real sy_cell[interp_order + 1];
                    const int k_cell = compute_shape_factor<interp_order>(sy_cell, ymid-0.5_rt);

                    amrex::Real coarse_value = 0.0_rt;
                    // sum interpolated contributions
                    for (int iy=0; iy<=interp_order; iy++){
                        for (int ix=0; ix<=interp_order; ix++){
                            coarse_value += sx_cell[ix]*sy_cell[iy]*
                                ((1.0_rt-rel_z)*arr_coarse(lo_coarse[0]+j_cell+ix,
                                                           lo_coarse[1]+k_cell+iy, lo_coarse[2])
                                   + rel_z*arr_coarse_prev(lo_coarse[0]+j_cell+ix,
                                                           lo_coarse[1]+k_cell+iy, lo_coarse[2]));
                        }
                    }

                    // set value on the fine grid to the interpolated value of the coarse grid
                    data_array(i,j,k) = coarse_value;
                }
            });
    }
}


void
Fields::SolvePoissonExmByAndEypBx (amrex::Vector<amrex::Geometry> const& geom,
                                   const MPI_Comm& m_comm_xy, const int lev, const int islice)
{
    /* Solves Laplacian(Psi) =  1/episilon0 * -(rho-Jz/c) and
     * calculates Ex-c By, Ey + c Bx from  grad(-Psi)
     */
    HIPACE_PROFILE("Fields::SolveExmByAndEypBx()");

    PhysConst phys_const = get_phys_const();

    // Left-Hand Side for Poisson equation is Psi in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Psi"], 1);

    InterpolateFromLev0toLev1(geom, lev, "rho", islice);

    // calculating the right-hand side 1/episilon0 * -(rho-Jz/c)
    FieldOperation(m_box_valid[lev], getStagingArea(lev),
                   1./(phys_const.c*phys_const.ep0), getField(lev, WhichSlice::This, "jz"),
                   -1./(phys_const.ep0), getField(lev, WhichSlice::This, "rho"),
                   m_box_problem[lev]);

    /*CopyToStagingArea(getSlices(lev,WhichSlice::This), SliceOperatorType::Assign,
                       Comps[WhichSlice::This]["jz"], lev);
    m_poisson_solver[lev]->StagingArea().mult(-1./phys_const.c);
    CopyToStagingArea(getSlices(lev,WhichSlice::This), SliceOperatorType::Add,
                       Comps[WhichSlice::This]["rho"], lev);
    m_poisson_solver[lev]->StagingArea().mult(-1./phys_const.ep0);*/

    InterpolateBoundaries(geom, lev, "Psi", islice);
    m_poisson_solver[lev]->SolvePoissonEquation(lhs);

    /* ---------- Transverse FillBoundary Psi ---------- */

    InterpolateFromLev0toLev1(geom, lev, "Psi", islice);

    /* Compute ExmBy and Eypbx from grad(-psi) */
    UnaryFieldOperation(m_box_valid[lev], getField(lev, WhichSlice::This, "ExmBy"),
                        -1., derivative_x{getField(lev, WhichSlice::This, "Psi"),
                        geom[lev].CellSize(Direction::x)});

    UnaryFieldOperation(m_box_valid[lev], getField(lev, WhichSlice::This, "EypBx"),
                        -1., derivative_y{getField(lev, WhichSlice::This, "Psi"),
                        geom[lev].CellSize(Direction::y)});

    /*TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        lev,
        -1.,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["Psi"],
        Comps[WhichSlice::This]["ExmBy"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        lev,
        -1.,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["Psi"],
        Comps[WhichSlice::This]["EypBx"]);*/
}


void
Fields::SolvePoissonEz (amrex::Vector<amrex::Geometry> const& geom, const int lev, const int islice)
{
    /* Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonEz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Ez"], 1);
    // Right-Hand Side for Poisson equation: compute 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
    // from the slice MF, and store in the staging area of poisson_solver
    FieldOperation(m_box_valid[lev], getStagingArea(lev),
                   1./(phys_const.ep0*phys_const.c),
                   derivative_x{getField(lev, WhichSlice::This, "jx"), geom[lev].CellSize(Direction::x)},
                   1./(phys_const.ep0*phys_const.c),
                   derivative_y{getField(lev, WhichSlice::This, "jy"), geom[lev].CellSize(Direction::y)},
                   m_box_source[lev]);

    /*TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        lev,
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jx"], 0, 1);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        lev,
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Add,
        Comps[WhichSlice::This]["jy"], 0, 1);*/

    InterpolateBoundaries(geom, lev, "Ez", islice);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(lhs);
}

void
Fields::SolvePoissonBx (amrex::MultiFab& Bx_iter, amrex::Vector<amrex::Geometry> const& geom,
                        const int lev, const int islice)
{
    /* Solves Laplacian(Bx) = mu_0*(- d_y(jz) + d_z(jy) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBx()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute -mu_0*d_y(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    FieldOperation(m_box_valid[lev], getStagingArea(lev),
                   -phys_const.mu0,
                   derivative_y{getField(lev, WhichSlice::This, "jz"), geom[lev].CellSize(Direction::y)},
                   phys_const.mu0,
                   derivative_z{getField(lev, WhichSlice::Previous1, "jy"),
                   getField(lev, WhichSlice::Next, "jy"), geom[lev].CellSize(Direction::z)},
                   m_box_source[lev]);

    /*TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        lev,
        -phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jz"], 0, 1);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver[lev]->StagingArea(),
        geom[lev].CellSize(Direction::z),
        lev,
        phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::Previous1]["jy"],
        Comps[WhichSlice::Next]["jy"]);*/

    InterpolateBoundaries(geom, lev, "Bx", islice);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(Bx_iter);
}

void
Fields::SolvePoissonBy (amrex::MultiFab& By_iter, amrex::Vector<amrex::Geometry> const& geom,
                        const int lev, const int islice)
{
    /* Solves Laplacian(By) = mu_0*(d_x(jz) - d_z(jx) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBy()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute mu_0*d_x(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    FieldOperation(m_box_valid[lev], getStagingArea(lev),
                   phys_const.mu0,
                   derivative_x{getField(lev, WhichSlice::This, "jz"), geom[lev].CellSize(Direction::x)},
                   -phys_const.mu0,
                   derivative_z{getField(lev, WhichSlice::Previous1, "jx"),
                   getField(lev, WhichSlice::Next, "jx"), geom[lev].CellSize(Direction::z)},
                   m_box_source[lev]);

    /*TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        lev,
        phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jz"], 0, 1);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver[lev]->StagingArea(),
        geom[lev].CellSize(Direction::z),
        lev,
        -phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::Previous1]["jx"],
        Comps[WhichSlice::Next]["jx"]);*/

    InterpolateBoundaries(geom, lev, "By", islice);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(By_iter);
}

void
Fields::SolvePoissonBz (amrex::Vector<amrex::Geometry> const& geom, const int lev, const int islice)
{
    /* Solves Laplacian(Bz) = mu_0*(d_y(jx) - d_x(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonBz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Bz"], 1);
    // Right-Hand Side for Poisson equation: compute mu_0*(d_y(jx) - d_x(jy))
    // from the slice MF, and store in the staging area of m_poisson_solver
    FieldOperation(m_box_valid[lev], getStagingArea(lev),
                   phys_const.mu0,
                   derivative_y{getField(lev, WhichSlice::This, "jx"), geom[lev].CellSize(Direction::y)},
                   -phys_const.mu0,
                   derivative_x{getField(lev, WhichSlice::This, "jy"), geom[lev].CellSize(Direction::x)},
                   m_box_source[lev]);

    /*TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        lev,
        phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jx"], 0, 1);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        lev,
        -phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::This]["jy"], 0, 1);*/

    InterpolateBoundaries(geom, lev, "Bz", islice);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(lhs);
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
