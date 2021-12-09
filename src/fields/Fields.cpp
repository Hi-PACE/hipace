#include "Fields.H"
#include "fft_poisson_solver/FFTPoissonSolverPeriodic.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichlet.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "particles/ShapeFactors.H"

amrex::IntVect Fields::m_slices_nguards = {-1, -1, -1};
amrex::IntVect Fields::m_poisson_nguards = {-1, -1, -1};

Fields::Fields (Hipace const* a_hipace)
    : m_slices(a_hipace->maxLevel()+1)
{
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

    // Need 1 extra guard cell transversally for transverse derivative
    int nguards_xy = std::max(1, Hipace::m_depos_order_xy);
    m_slices_nguards = {nguards_xy, nguards_xy, 0};
    m_poisson_nguards = {0, 0, 0};

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

// x = i * dx + GetPosOffset(0, geom, box);
// i = (x - GetPosOffset(0, geom, box))/dx;
amrex::Real GetPosOffset (const int direction, const amrex::Geometry& geom, const amrex::Box& box) {
    using namespace amrex::literals;
    return 0.5_rt*(geom.ProbLo(direction) + geom.ProbHi(direction)
           - geom.CellSize(direction) * (box.smallEnd(direction) + box.bigEnd(direction)));
}

template<int dir>
struct derivative_GPU {
    amrex::Array4<amrex::Real const> array;
    amrex::Real dx_inv;
    int box_lo;
    int box_hi;

    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j, int k) const noexcept {
        constexpr bool is_x_dir = dir == Direction::x;
        constexpr bool is_y_dir = dir == Direction::y;
        const int ij_along_dir = is_x_dir * i + is_y_dir * j;
        const bool lo_guard = ij_along_dir != box_lo;
        const bool hi_guard = ij_along_dir != box_hi;
        return (array(i+is_x_dir*hi_guard,j+is_y_dir*hi_guard,k)*hi_guard
               -array(i-is_x_dir*lo_guard,j-is_y_dir*lo_guard,k)*lo_guard) * dx_inv;
    }
};

template<>
struct derivative_GPU<Direction::z> {
    amrex::Array4<amrex::Real const> array1;
    amrex::Array4<amrex::Real const> array2;
    amrex::Real dz_inv;

    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j, int k) const noexcept {
        return (array1(i,j,k) - array2(i,j,k)) * dz_inv;
    }
};


template<int dir>
struct derivative {
    FieldView f_view;
    const amrex::Geometry& geom;

    derivative_GPU<dir> array (amrex::MFIter& mfi) const {
        amrex::Box bx = f_view.m_mfab[mfi].box();
        return derivative_GPU<dir>{f_view.array(mfi),
            1/(2*geom.CellSize(dir)), bx.smallEnd(dir), bx.bigEnd(dir)};
    }
};

template<>
struct derivative<Direction::z> {
    FieldView f_view1;
    FieldView f_view2;
    const amrex::Geometry& geom;

    derivative_GPU<Direction::z> array (amrex::MFIter& mfi) const {
        return derivative_GPU<Direction::z>{f_view1.array(mfi), f_view2.array(mfi),
            1/(2*geom.CellSize(Direction::z))};
    }
};

template<int interp_order_xy>
struct interpolated_field_GPU {
    amrex::Array4<amrex::Real const> arr_this;
    amrex::Array4<amrex::Real const> arr_prev;
    amrex::Real dx_inv;
    amrex::Real dy_inv;
    amrex::Real offset0;
    amrex::Real offset1;
    amrex::Real rel_z;
    int lo2;

    AMREX_GPU_DEVICE amrex::Real operator() (amrex::Real x, amrex::Real y) const noexcept {
        using namespace amrex::literals;

        // x direction
        const amrex::Real xmid = (x - offset0)*dx_inv;
        amrex::Real sx_cell[interp_order_xy + 1];
        const int i_cell = compute_shape_factor<interp_order_xy>(sx_cell, xmid);

        // y direction
        const amrex::Real ymid = (y - offset1)*dy_inv;
        amrex::Real sy_cell[interp_order_xy + 1];
        const int j_cell = compute_shape_factor<interp_order_xy>(sy_cell, ymid);

        amrex::Real field_value = 0.0_rt;
        // add interpolated contribution to boundary value
        for (int iy=0; iy<=interp_order_xy; iy++){
            for (int ix=0; ix<=interp_order_xy; ix++){
                field_value += sx_cell[ix]*sy_cell[iy]*
                    ((1.0_rt-rel_z)*arr_this(i_cell+ix,
                                             j_cell+iy, lo2)
                             +rel_z*arr_prev(i_cell+ix,
                                             j_cell+iy, lo2));
            }
        }
        return field_value;
    }
};

template<int interp_order_xy>
struct interpolated_field {
    FieldView f_view_this;
    FieldView f_view_prev;
    const amrex::Geometry& geom;
    amrex::Real rel_z;

    interpolated_field_GPU<interp_order_xy> array (amrex::MFIter& mfi) const {
        amrex::Box bx = f_view_this.m_mfab[mfi].box();
        return interpolated_field_GPU<interp_order_xy>{
            f_view_this.array(mfi), f_view_prev.array(mfi),
            1/geom.CellSize(0), 1/geom.CellSize(1),
            GetPosOffset(0, geom, bx), GetPosOffset(0, geom, bx),
            rel_z, bx.smallEnd(2)};
    }
};

template<class FVA, class FVB>
void
FieldOperation (const amrex::IntVect box_grow, FieldView dst,
                const amrex::Real factor_a, const FVA& src_a,
                const amrex::Real factor_b, const FVB& src_b)
{
    HIPACE_PROFILE("Fields::FieldOperation()");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(dst.m_mfab, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const auto dst_array = dst.array(mfi);
        const auto src_a_array = src_a.array(mfi);
        const auto src_b_array = src_b.array(mfi);
        const amrex::Box bx = mfi.growntilebox(box_grow);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                dst_array(i,j,k) = factor_a * src_a_array(i,j,k) + factor_b * src_b_array(i,j,k);
            });
    }
}

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
        slice_box -= amrex::IntVect(slice_box.smallEnd());
        if (!Diagnostic::m_include_ghost_cells) {
            slice_box -= m_slices_nguards;
        }
        slice_array = amrex::makeArray4(slice_fab.dataPtr(), slice_box, slice_fab.nComp());
        // slice_array's longitude index is 0.
    }

    const int full_array_z = i_slice / diag_coarsen[2];
    amrex::Box domain = geom.Domain();
    if (Diagnostic::m_include_ghost_cells) {
        domain.grow(m_slices_nguards);
    }
    const amrex::IntVect ncells_global = domain.length();

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

        const int cpyboxlo_x = Diagnostic::m_include_ghost_cells ? -m_slices_nguards[0] : 0;
        const int cpyboxlo_y = Diagnostic::m_include_ghost_cells ? -m_slices_nguards[1] : 0;

        const int *diag_comps = diag_comps_vect.data();

        amrex::ParallelFor(copy_box, ncomp,
        [=] AMREX_GPU_DEVICE (int i_l, int j_l, int k, int n) noexcept
        {
            const int m = n[diag_comps];
            const int i = i_l - cpyboxlo_x;
            const int j = j_l - cpyboxlo_y;

            // coarsening in slice direction is always 1
            const int i_c_start = amrex::min(i*coarse_x +(coarse_x-1)/2 -even_slice_x, ncells_x-1);
            const int i_c_stop  = amrex::min(i*coarse_x +coarse_x/2+1, ncells_x);
            const int j_c_start = amrex::min(j*coarse_y +(coarse_y-1)/2 -even_slice_y, ncells_y-1);
            const int j_c_stop  = amrex::min(j*coarse_y +coarse_y/2+1, ncells_y);

            amrex::Real field_value = 0._rt;
            int n_values = 0;

            for (int j_c = j_c_start; j_c != j_c_stop; ++j_c) {
                for (int i_c = i_c_start; i_c != i_c_stop; ++i_c) {
                    field_value += slice_array(i_c, j_c, 0, m+slice_comp);
                    ++n_values;
                }
            }

            full_array(i_l,j_l,k,n+full_comp) = field_value / amrex::max(n_values,1);
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
                             Comps[WhichSlice::RhoIons]["rho"], Comps[WhichSlice::This]["rho"], 1,
                             m_slices_nguards);
    } else {
        amrex::MultiFab::Subtract(getSlices(lev, WhichSlice::This), getSlices(lev, WhichSlice::RhoIons),
                                  Comps[WhichSlice::RhoIons]["rho"], Comps[WhichSlice::This]["rho"], 1,
                                  m_slices_nguards);
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



template<class Functional>
void
SetDirichletBoundaries (amrex::Array4<amrex::Real> dst, const amrex::Box& solver_size,
                        const amrex::Geometry& geom, const Functional& boundary_value)
{
    const int box_len0 = solver_size.length(0);
    const int box_len1 = solver_size.length(1);
    const int box_lo0 = solver_size.smallEnd(0);
    const int box_lo1 = solver_size.smallEnd(1);
    const int box_lo2 = solver_size.smallEnd(2);
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);
    const amrex::Real offset0 = GetPosOffset(0, geom, solver_size);
    const amrex::Real offset1 = GetPosOffset(1, geom, solver_size);

    const amrex::Box edge_box = {{0, 0, 0}, {box_len0 + box_len1 - 1, 1, 0}};

    amrex::ParallelFor(edge_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            const bool i_is_changing = (i < box_len0);
            const bool i_lo_edge = (!i_is_changing) && (!j);
            const bool i_hi_edge = (!i_is_changing) && j;
            const bool j_lo_edge = i_is_changing && (!j);
            const bool j_hi_edge = i_is_changing && j;

            const int i_idx = box_lo0 + i_hi_edge*(box_len0-1) + i_is_changing*i;
            const int j_idx = box_lo1 + j_hi_edge*(box_len1-1) + (!i_is_changing)*(i-box_len0);

            const int i_idx_offset = i_idx - i_lo_edge + i_hi_edge;
            const int j_idx_offset = j_idx - j_lo_edge + j_hi_edge;

            const amrex::Real x = i_idx_offset * dx + offset0;
            const amrex::Real y = j_idx_offset * dy + offset1;

            const amrex::Real dxdx = dx*dx*(!i_is_changing) + dy*dy*i_is_changing;

            amrex::Gpu::Atomic::AddNoRet(&(dst(i_idx, j_idx, box_lo2)),
                                         - boundary_value(x, y) / dxdx);
        });
}

void
Fields::SetRefinedBoundaries (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                              std::string component, const int islice)
{
    HIPACE_PROFILE("Fields::SetRefinedBoundaries()");
    if (lev == 0) return; // only interpolate boundaries to lev 1
    constexpr int interp_order = 2;

    const amrex::Real ref_ratio_z = geom[lev-1].CellSize(2) / geom[lev].CellSize(2);
    const amrex::Real islice_coarse_real = islice / ref_ratio_z;
    const int islice_coarse_int = islice_coarse_real;
    const amrex::Real rel_z = islice_coarse_real - islice_coarse_int;

    auto solution_interp = interpolated_field<interp_order>{
        getField(lev-1, WhichSlice::This, component),
        getField(lev-1, WhichSlice::Previous1, component),
        geom[lev-1], rel_z};
    FieldView staging_area = getStagingArea(lev);

    for (amrex::MFIter mfi(staging_area.m_mfab, false); mfi.isValid(); ++mfi)
    {
        const auto arr_solution_interp = solution_interp.array(mfi);
        const auto arr_staging_area = staging_area.array(mfi);
        const amrex::Box fine_staging_box = staging_area.m_mfab[mfi].box();

        SetDirichletBoundaries(arr_staging_area, fine_staging_box, geom[lev], arr_solution_interp);
    }
}


void
Fields::InterpolateFromLev0toLev1 (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                                   std::string component, const int islice,
                                   const amrex::IntVect outer_edge, const amrex::IntVect inner_edge)
{
    if (lev == 0) return; // only interpolate boundaries to lev 1
    constexpr int interp_order = 2;
    if (outer_edge == inner_edge) return;

    const amrex::Real ref_ratio_z = geom[lev-1].CellSize(2) / geom[lev].CellSize(2);
    const amrex::Real islice_coarse_real = islice / ref_ratio_z;
    const int islice_coarse_int = islice_coarse_real;
    const amrex::Real rel_z = islice_coarse_real - islice_coarse_int;

    auto field_coarse_interp = interpolated_field<interp_order>{
        getField(lev-1, WhichSlice::This, component),
        getField(lev-1, WhichSlice::Previous1, component),
        geom[lev-1], rel_z};
    FieldView field_fine = getField(lev, WhichSlice::This, component);

    for (amrex::MFIter mfi( field_fine.m_mfab, false); mfi.isValid(); ++mfi)
    {
        auto arr_field_coarse_interp = field_coarse_interp.array(mfi);
        auto arr_field_fine = field_fine.array(mfi);
        const amrex::Box fine_box_extended = mfi.growntilebox(outer_edge);
        const amrex::Box fine_box_narrow = mfi.growntilebox(inner_edge);

        const int narrow_i_lo = fine_box_narrow.smallEnd(0);
        const int narrow_i_hi = fine_box_narrow.bigEnd(0);
        const int narrow_j_lo = fine_box_narrow.smallEnd(1);
        const int narrow_j_hi = fine_box_narrow.bigEnd(1);

        const amrex::Real dx = geom[lev].CellSize(0);
        const amrex::Real dy = geom[lev].CellSize(1);
        const amrex::Real offset0 = GetPosOffset(0, geom[lev], fine_box_extended);
        const amrex::Real offset1 = GetPosOffset(1, geom[lev], fine_box_extended);

        amrex::ParallelFor(fine_box_extended,
            [=] AMREX_GPU_DEVICE (int i, int j , int k) noexcept
            {
                if(i<narrow_i_lo || i>narrow_i_hi || j<narrow_j_lo || j>narrow_j_hi) {
                    amrex::Real x = i * dx + offset0;
                    amrex::Real y = j * dy + offset1;
                    arr_field_fine(i,j,k) = arr_field_coarse_interp(x,y);
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

    InterpolateFromLev0toLev1(geom, lev, "rho", islice, m_poisson_nguards, -m_slices_nguards);

    // calculating the right-hand side 1/episilon0 * -(rho-Jz/c)
    FieldOperation(m_poisson_nguards, getStagingArea(lev),
                   1./(phys_const.c*phys_const.ep0), getField(lev, WhichSlice::This, "jz"),
                   -1./(phys_const.ep0), getField(lev, WhichSlice::This, "rho"));

    SetRefinedBoundaries(geom, lev, "Psi", islice);
    m_poisson_solver[lev]->SolvePoissonEquation(lhs);

    /* ---------- Transverse FillBoundary Psi ---------- */
    amrex::ParallelContext::push(m_comm_xy);
    lhs.FillBoundary(geom[lev].periodicity());
    amrex::ParallelContext::pop();

    InterpolateFromLev0toLev1(geom, lev, "Psi", islice, m_slices_nguards, m_poisson_nguards);

    /* Compute ExmBy and Eypbx from grad(-psi) */
    FieldView f_ExmBy = getField(lev, WhichSlice::This, "ExmBy");
    FieldView f_EypBx = getField(lev, WhichSlice::This, "EypBx");
    FieldView f_Psi = getField(lev, WhichSlice::This, "Psi");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(f_ExmBy.m_mfab, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Array4<amrex::Real> array_ExmBy = f_ExmBy.array(mfi);
        const amrex::Array4<amrex::Real> array_EypBx = f_EypBx.array(mfi);
        const amrex::Array4<amrex::Real const> array_Psi = f_Psi.array(mfi);
        const amrex::Box bx = mfi.growntilebox(m_slices_nguards - amrex::IntVect{1, 1, 0});
        const amrex::Real dx_inv = 1./(2*geom[lev].CellSize(Direction::x));
        const amrex::Real dy_inv = 1./(2*geom[lev].CellSize(Direction::y));

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                array_ExmBy(i,j,k) = - (array_Psi(i+1,j,k) - array_Psi(i-1,j,k))*dx_inv;
                array_EypBx(i,j,k) = - (array_Psi(i,j+1,k) - array_Psi(i,j-1,k))*dy_inv;
            });
    }
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
    FieldOperation(m_poisson_nguards, getStagingArea(lev),
                   1./(phys_const.ep0*phys_const.c),
                   derivative<Direction::x>{getField(lev, WhichSlice::This, "jx"), geom[lev]},
                   1./(phys_const.ep0*phys_const.c),
                   derivative<Direction::y>{getField(lev, WhichSlice::This, "jy"), geom[lev]});


    SetRefinedBoundaries(geom, lev, "Ez", islice);
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
    FieldOperation(m_poisson_nguards, getStagingArea(lev),
                   -phys_const.mu0,
                   derivative<Direction::y>{getField(lev, WhichSlice::This, "jz"), geom[lev]},
                   phys_const.mu0,
                   derivative<Direction::z>{getField(lev, WhichSlice::Previous1, "jy"),
                   getField(lev, WhichSlice::Next, "jy"), geom[lev]});


    SetRefinedBoundaries(geom, lev, "Bx", islice);
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
    FieldOperation(m_poisson_nguards, getStagingArea(lev),
                   phys_const.mu0,
                   derivative<Direction::x>{getField(lev, WhichSlice::This, "jz"), geom[lev]},
                   -phys_const.mu0,
                   derivative<Direction::z>{getField(lev, WhichSlice::Previous1, "jx"),
                   getField(lev, WhichSlice::Next, "jx"), geom[lev]});


    SetRefinedBoundaries(geom, lev, "By", islice);
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
    FieldOperation(m_poisson_nguards, getStagingArea(lev),
                   phys_const.mu0,
                   derivative<Direction::y>{getField(lev, WhichSlice::This, "jx"), geom[lev]},
                   -phys_const.mu0,
                   derivative<Direction::x>{getField(lev, WhichSlice::This, "jy"), geom[lev]});


    SetRefinedBoundaries(geom, lev, "Bz", islice);
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
        Comps[WhichSlice::This]["Bx"], 1, m_slices_nguards);

    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1+mix_factor_init_guess, getSlices(lev, WhichSlice::Previous1), Comps[WhichSlice::Previous1]["By"],
        -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2), Comps[WhichSlice::Previous2]["By"],
        Comps[WhichSlice::This]["By"], 1, m_slices_nguards);
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
        0, 1, m_slices_nguards);

    /* calculating the mixed B field  B = a*B + (1-a)*B_prev_iter */
    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1-predcorr_B_mixing_factor, getSlices(lev, WhichSlice::This), field_comp,
        predcorr_B_mixing_factor, B_prev_iter, 0,
        field_comp, 1, m_slices_nguards);

    /* Shifting the B field from the current iteration to the previous iteration */
    amrex::MultiFab::Copy(B_prev_iter, B_iter, 0, 0, 1, m_slices_nguards);

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
