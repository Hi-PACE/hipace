/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs, WeiqunZhang
 * coulibaly-mouhamed
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "fft_poisson_solver/FFTPoissonSolverPeriodic.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletDirect.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletExpanded.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletFast.H"
#include "fft_poisson_solver/MGPoissonSolverDirichlet.H"
#include "Hipace.H"
#include "OpenBoundary.H"
#include "utils/DeprecatedInput.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/InsituUtil.H"
#include "particles/particles_utils/ShapeFactors.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif

using namespace amrex::literals;

Fields::Fields (const int nlev)
    : m_slices(nlev)
{
    amrex::ParmParse ppf("fields");
    DeprecatedInput("fields", "do_dirichlet_poisson", "poisson_solver", "");
    // set default Poisson solver based on the platform
#ifdef AMREX_USE_GPU
    m_poisson_solver_str = "FFTDirichletFast";
#else
    m_poisson_solver_str = "FFTDirichletDirect";
#endif
    queryWithParser(ppf, "poisson_solver", m_poisson_solver_str);
    queryWithParser(ppf, "insitu_period", m_insitu_period);
    queryWithParser(ppf, "insitu_file_prefix", m_insitu_file_prefix);
    queryWithParser(ppf, "do_symmetrize", m_do_symmetrize);
    DeprecatedInput("fields", "extended_solve",
                    "boundary.particle_lo and boundary.particle_hi", "", true);
    DeprecatedInput("fields", "open_boundary", "boundary.field = Open", "", true);
}

void
Fields::AllocData (
    int lev, amrex::Geometry const& geom, const amrex::BoxArray& slice_ba,
    const amrex::DistributionMapping& slice_dm)
{
    HIPACE_PROFILE("Fields::AllocData()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(slice_ba.size() == 1,
        "Parallel field solvers not supported yet");

    if (lev==0) {

        m_lev0_periodicity = geom.periodicity();

        // Need 1 extra guard cell transversally for transverse derivative
        int nguards_xy = (Hipace::m_depos_order_xy + 1) / 2 + 1;
        m_slices_nguards = amrex::IntVect{nguards_xy, nguards_xy, 0};

        m_explicit = Hipace::m_explicit;
        m_any_neutral_background = Hipace::GetInstance().m_multi_plasma.AnySpeciesNeutralizeBackground();
        const bool any_salame = Hipace::GetInstance().m_multi_beam.AnySpeciesSalame();

        if (m_explicit) {
            // explicit solver:
            // beams share jx_beam jy_beam jz_beam
            // jx jy rhomjz for all plasmas and beams
            // rho is plasma-only if used

            int isl = WhichSlice::Next;
            Comps[isl].multi_emplace(N_Comps, "jx_beam", "jy_beam");

            isl = WhichSlice::This;
            // (Bx, By), (Sy, Sx) and (chi, chi2) adjacent for explicit solver
            Comps[isl].multi_emplace(N_Comps, "chi");
            if (Hipace::m_use_amrex_mlmg) {
                Comps[isl].multi_emplace(N_Comps, "chi2");
            }
            Comps[isl].multi_emplace(N_Comps, "Sy", "Sx", "ExmBy", "EypBx", "Ez",
                "Bx", "By", "Bz", "Psi",
                "jx_beam", "jy_beam", "jz_beam", "jx", "jy", "rhomjz");
            if (Hipace::m_use_laser) {
                Comps[isl].multi_emplace(N_Comps, "aabs");
            }
            if (Hipace::m_deposit_rho) {
                Comps[isl].multi_emplace(N_Comps, "rho");
            }
            if (Hipace::m_deposit_rho_individual) {
                for (auto& plasma_name : Hipace::GetInstance().m_multi_plasma.GetNames()) {
                    Comps[isl].multi_emplace(N_Comps, "rho_" + plasma_name);
                }
            }
            if (Hipace::m_do_beam_jz_minus_rho) {
                Comps[isl].multi_emplace(N_Comps, "rhomjz_beam");
            }

            isl = WhichSlice::Previous;
            Comps[isl].multi_emplace(N_Comps, "jx_beam", "jy_beam");

            isl = WhichSlice::RhomJzIons;
            if (m_any_neutral_background) {
                Comps[isl].multi_emplace(N_Comps, "rhomjz");
            }

            isl = WhichSlice::Salame;
            if (any_salame) {
                Comps[isl].multi_emplace(N_Comps, "Ez_target", "Ez_no_salame", "Ez",
                    "jx", "jy", "jz_beam", "Bx", "By", "Sy", "Sx", "Sy_back", "Sx_back");
            }

            isl = WhichSlice::PCIter;
            // empty

            isl = WhichSlice::PCPrevIter;
            // empty

        } else {
            // predictor-corrector:
            // all beams and plasmas share jx jy jz rhomjz
            // rho is plasma-only if used

            int isl = WhichSlice::Next;
            Comps[isl].multi_emplace(N_Comps, "jx", "jy");

            isl = WhichSlice::This;
            // Bx and By adjacent for explicit solver
            Comps[isl].multi_emplace(N_Comps, "ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "Psi",
                                              "jx", "jy", "jz", "rhomjz");

            if (Hipace::m_use_laser) {
                Comps[isl].multi_emplace(N_Comps, "chi", "aabs");
            }
            if (Hipace::m_deposit_rho) {
                Comps[isl].multi_emplace(N_Comps, "rho");
            }
            if (Hipace::m_deposit_rho_individual) {
                for (auto& plasma_name : Hipace::GetInstance().m_multi_plasma.GetNames()) {
                    Comps[isl].multi_emplace(N_Comps, "rho_" + plasma_name);
                }
            }

            isl = WhichSlice::Previous;
            Comps[isl].multi_emplace(N_Comps, "Bx", "By", "jx", "jy");


            isl = WhichSlice::RhomJzIons;
            if (m_any_neutral_background) {
                Comps[isl].multi_emplace(N_Comps, "rhomjz");
            }

            isl = WhichSlice::Salame;
            // empty, not compatible

            isl = WhichSlice::PCIter;
            Comps[isl].multi_emplace(N_Comps, "Bx", "By");

            isl = WhichSlice::PCPrevIter;
            Comps[isl].multi_emplace(N_Comps, "Bx", "By");
        }
    }

    // allocate memory for fields
    if (N_Comps != 0) {
        m_slices[lev].define(
            slice_ba, slice_dm, N_Comps, m_slices_nguards,
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[lev].setVal(0._rt);
    }

    // The Poisson solver operates on transverse slices only.
    // The constructor takes the BoxArray and the DistributionMap of a slice,
    // so the FFTPlans are built on a slice.
    if (m_poisson_solver_str == "FFTDirichletDirect"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletDirect>(
            new FFTPoissonSolverDirichletDirect(getSlices(lev).boxArray(),
                                                getSlices(lev).DistributionMap(),
                                                geom)) );
    } else if (m_poisson_solver_str == "FFTDirichletExpanded"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletExpanded>(
            new FFTPoissonSolverDirichletExpanded(getSlices(lev).boxArray(),
                                                  getSlices(lev).DistributionMap(),
                                                  geom)) );
    } else if (m_poisson_solver_str == "FFTDirichletFast"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletFast>(
            new FFTPoissonSolverDirichletFast(getSlices(lev).boxArray(),
                                              getSlices(lev).DistributionMap(),
                                              geom)) );
    } else if (m_poisson_solver_str == "FFTPeriodic") {
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverPeriodic>(
            new FFTPoissonSolverPeriodic(getSlices(lev).boxArray(),
                                         getSlices(lev).DistributionMap(),
                                         geom))  );
    } else if (m_poisson_solver_str == "MGDirichlet") {
        m_poisson_solver.push_back(std::unique_ptr<MGPoissonSolverDirichlet>(
            new MGPoissonSolverDirichlet(getSlices(lev).boxArray(),
                                         getSlices(lev).DistributionMap(),
                                         geom))  );
    } else {
        amrex::Abort("Unknown poisson solver '" + m_poisson_solver_str +
            "', must be 'FFTDirichletDirect', 'FFTDirichletExpanded', 'FFTDirichletFast', " +
            "'FFTPeriodic' or 'MGDirichlet'");
    }

    if (lev == 0 && m_insitu_period > 0) {
#ifdef HIPACE_USE_OPENPMD
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_insitu_file_prefix !=
            Hipace::GetInstance().m_openpmd_writer.m_file_prefix,
            "Must choose a different field insitu file prefix compared to the full diagnostics");
#endif
        // Allocate memory for in-situ diagnostics
        m_insitu_rdata.resize(geom.Domain().length(2)*m_insitu_nrp, 0.);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
    }
}

/** \brief inner version of derivative */
template<int dir>
struct derivative_inner {
    // captured variables for GPU
    Array2<amrex::Real const> array;
    amrex::Real dx_inv;

    // derivative of field in dir direction (x or y)
    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j) const noexcept {
        constexpr bool is_x_dir = dir == Direction::x;
        constexpr bool is_y_dir = dir == Direction::y;
        return (array(i+is_x_dir,j+is_y_dir) - array(i-is_x_dir,j-is_y_dir)) * dx_inv;
    }
};

/** \brief inner version of derivative */
template<>
struct derivative_inner<Direction::z> {
    // captured variables for GPU
    Array2<amrex::Real const> array1;
    Array2<amrex::Real const> array2;
    amrex::Real dz_inv;

    // derivative of field in z direction
    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j) const noexcept {
        return (array1(i,j) - array2(i,j)) * dz_inv;
    }
};

/** \brief derivative in x or y direction */
template<int dir>
struct derivative {
    // use brace initialization as constructor
    amrex::MultiFab f_view; // field to calculate its derivative
    const amrex::Geometry& geom; // geometry of field

    // use .array(mfi) like with amrex::MultiFab
    derivative_inner<dir> array (amrex::MFIter& mfi) const {
        return derivative_inner<dir>{f_view.array(mfi), 0.5_rt*geom.InvCellSize(dir)};
    }
};

/** \brief derivative in z direction. Use fields from previous and next slice */
template<>
struct derivative<Direction::z> {
    // use brace initialization as constructor
    amrex::MultiFab f_view1; // field on previous slice to calculate its derivative
    amrex::MultiFab f_view2; // field on next slice to calculate its derivative
    const amrex::Geometry& geom; // geometry of field

    // use .array(mfi) like with amrex::MultiFab
    derivative_inner<Direction::z> array (amrex::MFIter& mfi) const {
        return derivative_inner<Direction::z>{f_view1.array(mfi), f_view2.array(mfi),
            0.5_rt*geom.InvCellSize(Direction::z)};
    }
};

/** \brief inner version of interpolated_field_xy */
template<int interp_order_xy, class ArrayType>
struct interpolated_field_xy_inner {
    // captured variables for GPU
    ArrayType array;
    amrex::Real dx_inv;
    amrex::Real dy_inv;
    amrex::Real offset0;
    amrex::Real offset1;

    // interpolate field in x, y with interp_order_xy order transversely,
    // x and y must be inside field box
    template<class...Args> AMREX_GPU_DEVICE
    amrex::Real operator() (amrex::Real x, amrex::Real y, Args...args) const noexcept {

        // x direction
        const amrex::Real xmid = (x - offset0)*dx_inv;
        amrex::Real sx_cell[interp_order_xy + 1];
        const int i_cell = compute_shape_factor<interp_order_xy>(sx_cell, xmid);

        // y direction
        const amrex::Real ymid = (y - offset1)*dy_inv;
        amrex::Real sy_cell[interp_order_xy + 1];
        const int j_cell = compute_shape_factor<interp_order_xy>(sy_cell, ymid);

        amrex::Real field_value = 0._rt;
        for (int iy=0; iy<=interp_order_xy; iy++){
            for (int ix=0; ix<=interp_order_xy; ix++){
                field_value += sx_cell[ix] * sy_cell[iy] * array(i_cell+ix, j_cell+iy, args...);
            }
        }
        return field_value;
    }
};

/** \brief interpolate field in x, y with interp_order_xy order transversely,
 * x and y must be inside field box */
template<int interp_order_xy, class MfabType>
struct interpolated_field_xy {
    // use brace initialization as constructor
    MfabType mfab; // MultiFab type object of the field
    amrex::Geometry geom; // geometry of field

    // use .array(mfi) like with amrex::MultiFab
    auto array (amrex::MFIter& mfi) const {
        auto mfab_array = to_array2(mfab.array(mfi));
        return interpolated_field_xy_inner<interp_order_xy, decltype(mfab_array)>{
            mfab_array, geom.InvCellSize(0), geom.InvCellSize(1),
            GetPosOffset(0, geom, geom.Domain()), GetPosOffset(1, geom, geom.Domain())};
    }
};

/** \brief inner version of guarded_field_xy */
struct guarded_field_xy_inner {
    // captured variables for GPU
    Array3<amrex::Real const> array;
    int lox;
    int hix;
    int loy;
    int hiy;

    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j, int n) const noexcept {
        if (lox <= i && i <= hix && loy <= j && j <= hiy) {
            return array(i,j,n);
        } else return 0._rt;
    }
};

/** \brief if indices are outside of the fields box zero is returned */
struct guarded_field_xy {
    // use brace initialization as constructor
    amrex::MultiFab& mfab; // field to be guarded (zero extended)

    // use .array(mfi) like with amrex::MultiFab
    guarded_field_xy_inner array (amrex::MFIter& mfi) const {
        const amrex::Box bx = mfab[mfi].box();
        return guarded_field_xy_inner{mfab.const_array(mfi), bx.smallEnd(Direction::x),
            bx.bigEnd(Direction::x), bx.smallEnd(Direction::y), bx.bigEnd(Direction::y)};
    }
};

/** \brief Calculates dst = factor_a*src_a + factor_b*src_b. src_a and src_b can be derivatives
 *
 * \param[in] dst destination
 * \param[in] factor_a factor before src_a
 * \param[in] src_a first source
 * \param[in] factor_b factor before src_b
 * \param[in] src_b second source
 */
template<class FVA, class FVB>
void
LinCombination (amrex::MultiFab dst,
                const amrex::Real factor_a, const FVA& src_a,
                const amrex::Real factor_b, const FVB& src_b)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(dst, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const Array2<amrex::Real> dst_array = dst.array(mfi);
        const auto src_a_array = to_array2(src_a.array(mfi));
        const auto src_b_array = to_array2(src_b.array(mfi));
        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                dst_array(i,j) = factor_a * src_a_array(i,j) + factor_b * src_b_array(i,j);
            });
    }
}

/** \brief Calculates dst = factor*src. src can be a derivative
 *
 * \param[in] dst destination
 * \param[in] factor factor before src_a
 * \param[in] src first source
 */
template<class FV>
void
Multiply (amrex::MultiFab dst, const amrex::Real factor, const FV& src)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(dst, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const Array2<amrex::Real> dst_array = dst.array(mfi);
        const auto src_array = to_array2(src.array(mfi));
        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                dst_array(i,j) = factor * src_array(i,j);
            });
    }
}

void
Fields::Copy (const int current_N_level, const int i_slice, FieldDiagnosticData& fd,
              const amrex::Vector<amrex::Geometry>& field_geom, MultiLaser& multi_laser)
{
    HIPACE_PROFILE("Fields::Copy()");
    constexpr int depos_order_xy = 1;
    constexpr int depos_order_z = 1;
    constexpr int depos_order_offset = depos_order_z / 2 + 1;

    const amrex::Real poff_calc_z = GetPosOffset(2, field_geom[0], field_geom[0].Domain());
    const amrex::Real poff_diag_x = GetPosOffset(0, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real poff_diag_y = GetPosOffset(1, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real poff_diag_z = GetPosOffset(2, fd.m_geom_io, fd.m_geom_io.Domain());

    // Interpolation in z Direction, done as if looped over diag_fab not i_slice
    // Calculate to which diag_fab slices this slice could contribute
    const int i_slice_min = i_slice - depos_order_offset;
    const int i_slice_max = i_slice + depos_order_offset;
    const amrex::Real pos_slice_min = i_slice_min * field_geom[0].CellSize(2) + poff_calc_z;
    const amrex::Real pos_slice_max = i_slice_max * field_geom[0].CellSize(2) + poff_calc_z;
    int k_min = static_cast<int>(amrex::Math::round((pos_slice_min - poff_diag_z)
                                                          * fd.m_geom_io.InvCellSize(2)));
    const int k_max = static_cast<int>(amrex::Math::round((pos_slice_max - poff_diag_z)
                                                          * fd.m_geom_io.InvCellSize(2)));

    amrex::Box diag_box = fd.m_geom_io.Domain();
    if (fd.m_slice_dir != 2) {
        // Put contributions from i_slice to different diag_fab slices in GPU vector
        m_rel_z_vec.resize(k_max+1-k_min);
        m_rel_z_vec_cpu.resize(k_max+1-k_min);
        for (int k=k_min; k<=k_max; ++k) {
            const amrex::Real pos = k * fd.m_geom_io.CellSize(2) + poff_diag_z;
            const amrex::Real mid_i_slice = (pos - poff_calc_z)*field_geom[0].InvCellSize(2);
            amrex::Real sz_cell[depos_order_z + 1];
            const int k_cell = compute_shape_factor<depos_order_z>(sz_cell, mid_i_slice);
            m_rel_z_vec_cpu[k-k_min] = 0;
            for (int i=0; i<=depos_order_z; ++i) {
                if (k_cell+i == i_slice) {
                    m_rel_z_vec_cpu[k-k_min] = sz_cell[i];
                }
            }
        }

        // Optimization: don’t loop over diag_fab slices with 0 contribution
        int k_start = k_min;
        int k_stop = k_max;
        for (int k=k_min; k<=k_max; ++k) {
            if (m_rel_z_vec_cpu[k-k_min] == 0) ++k_start;
            else break;
        }
        for (int k=k_max; k>=k_min; --k) {
            if (m_rel_z_vec_cpu[k-k_min] == 0) --k_stop;
            else break;
        }
        diag_box.setSmall(2, amrex::max(diag_box.smallEnd(2), k_start));
        diag_box.setBig(2, amrex::min(diag_box.bigEnd(2), k_stop));
    } else {
        m_rel_z_vec.resize(1);
        m_rel_z_vec_cpu.resize(1);
        const amrex::Real pos_z = i_slice * field_geom[0].CellSize(2) + poff_calc_z;
        if (fd.m_geom_io.ProbLo(2) <= pos_z && pos_z <= fd.m_geom_io.ProbHi(2)) {
            m_rel_z_vec_cpu[0] = field_geom[0].CellSize(2);
            k_min = 0;
        } else {
            return;
        }
    }
    if (diag_box.isEmpty()) return;
    auto& slice_mf = m_slices[fd.m_level];
    auto slice_func = interpolated_field_xy<depos_order_xy,
        guarded_field_xy>{{slice_mf}, field_geom[fd.m_level]};
    auto& laser_mf = multi_laser.getSlices();
    auto laser_func = interpolated_field_xy<depos_order_xy,
        guarded_field_xy>{{laser_mf}, multi_laser.GetLaserGeom()};

#ifdef AMREX_USE_GPU
    // This async copy happens on the same stream as the ParallelFor below, which uses the copied array.
    // Therefore, it is safe to do it async.
    amrex::Gpu::htod_memcpy_async(m_rel_z_vec.dataPtr(), m_rel_z_vec_cpu.dataPtr(),
                                  m_rel_z_vec_cpu.size() * sizeof(amrex::Real));
#else
    std::memcpy(m_rel_z_vec.dataPtr(), m_rel_z_vec_cpu.dataPtr(),
                m_rel_z_vec_cpu.size() * sizeof(amrex::Real));
#endif

    // Finally actual kernel: Interpolation in x, y, z of zero-extended fields
    for (amrex::MFIter mfi(slice_mf, DfltMfi); mfi.isValid(); ++mfi) {
        const int *diag_comps = fd.m_comps_output_idx.data();
        const amrex::Real *rel_z_data = m_rel_z_vec.data();
        const amrex::Real dx = fd.m_geom_io.CellSize(0);
        const amrex::Real dy = fd.m_geom_io.CellSize(1);

        if (fd.m_base_geom_type == FieldDiagnosticData::geom_type::field &&
            current_N_level > fd.m_level) {
            auto slice_array = slice_func.array(mfi);
            amrex::Array4<amrex::Real> diag_array = fd.m_F.array();
            amrex::ParallelFor(diag_box, fd.m_nfields,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
                {
                    const amrex::Real x = i * dx + poff_diag_x;
                    const amrex::Real y = j * dy + poff_diag_y;
                    const int m = n[diag_comps];
                    diag_array(i,j,k,n) += rel_z_data[k-k_min] * slice_array(x,y,m);
                });
        } else if (fd.m_base_geom_type == FieldDiagnosticData::geom_type::laser &&
                   multi_laser.UseLaser(i_slice)) {
            auto laser_array = laser_func.array(mfi);
            amrex::Array4<amrex::GpuComplex<amrex::Real>> diag_array_laser = fd.m_F_laser.array();
            amrex::ParallelFor(diag_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const amrex::Real x = i * dx + poff_diag_x;
                    const amrex::Real y = j * dy + poff_diag_y;
                    diag_array_laser(i,j,k) += amrex::GpuComplex<amrex::Real> {
                        rel_z_data[k-k_min] * laser_array(x,y,WhichLaserSlice::n00j00_r),
                        rel_z_data[k-k_min] * laser_array(x,y,WhichLaserSlice::n00j00_i)
                    };
                });
        }
    }
}

void
Fields::InitializeSlices (int lev, int islice, const amrex::Vector<amrex::Geometry>& geom)
{
    HIPACE_PROFILE("Fields::InitializeSlices()");

    if (Hipace::m_explicit) {
        if (lev != 0 && islice == geom[lev].Domain().bigEnd(Direction::z)) {
            // first slice of lev (islice goes backwards)
            // iterpolate jx_beam and jy_beam from lev-1 to lev
            LevelUp(geom, lev, WhichSlice::Previous, "jx_beam");
            LevelUp(geom, lev, WhichSlice::Previous, "jy_beam");
            LevelUp(geom, lev, WhichSlice::This, "jx_beam");
            LevelUp(geom, lev, WhichSlice::This, "jy_beam");
            duplicate(lev, WhichSlice::This, {"jx"     , "jy"     },
                           WhichSlice::This, {"jx_beam", "jy_beam"});
        }
        // Set all quantities on WhichSlice::This to 0 except:
        // Bx, By, Bz, Psi and Ez which are set by field solvers and
        // jx, jy, jx_beam and jy_beam on WhichSlice::This:
        // shifted from the previous WhichSlice::Next
        // with jx and jy initially set to jx_beam and jy_beam
        setVal(0., lev, WhichSlice::This, "chi", "Sy", "Sx", "ExmBy", "EypBx", "jz_beam", "rhomjz");
        setVal(0., lev, WhichSlice::Next, "jx_beam", "jy_beam");
        if (Hipace::m_do_beam_jz_minus_rho) {
            setVal(0., lev, WhichSlice::This, "rhomjz_beam");
        }
    } else {
        if (lev != 0 && islice == geom[lev].Domain().bigEnd(Direction::z)) {
            // first slice of lev (islice goes backwards)
            // iterpolate Bx, By, jx and jy from lev-1 to lev
            LevelUp(geom, lev, WhichSlice::PCPrevIter, "Bx");
            LevelUp(geom, lev, WhichSlice::PCPrevIter, "By");
            LevelUp(geom, lev, WhichSlice::Previous, "Bx");
            LevelUp(geom, lev, WhichSlice::Previous, "By");
            LevelUp(geom, lev, WhichSlice::Previous, "jx");
            LevelUp(geom, lev, WhichSlice::Previous, "jy");
        }
        setVal(0., lev, WhichSlice::This,
            "ExmBy", "EypBx", "jx", "jy", "jz", "rhomjz");
        if (Hipace::m_use_laser) {
            setVal(0., lev, WhichSlice::This, "chi");
        }
    }
    if (Hipace::m_deposit_rho) {
        setVal(0., lev, WhichSlice::This, "rho");
    }
    if (Hipace::m_deposit_rho_individual) {
        for (auto& plasma_name : Hipace::GetInstance().m_multi_plasma.GetNames()) {
            setVal(0., lev, WhichSlice::This, "rho_" + plasma_name);
        }
    }
}

void
Fields::ShiftSlices (int lev)
{
    HIPACE_PROFILE("Fields::ShiftSlices()");

    const bool explicit_solve = Hipace::m_explicit;

    // only shift the slices that are allocated
    if (explicit_solve) {
        shift(lev, WhichSlice::Previous, WhichSlice::This, "jx_beam", "jy_beam");
        duplicate(lev, WhichSlice::This, {"jx_beam", "jy_beam", "jx"     , "jy"     },
                       WhichSlice::Next, {"jx_beam", "jy_beam", "jx_beam", "jy_beam"});
    } else {
        shift(lev, WhichSlice::PCPrevIter, WhichSlice::Previous, "Bx", "By");
        shift(lev, WhichSlice::Previous, WhichSlice::This, "Bx", "By", "jx", "jy");
    }
}

void
Fields::AddRhoIons (const int lev)
{
    if (!m_any_neutral_background) return;
    HIPACE_PROFILE("Fields::AddRhoIons()");
    add(lev, WhichSlice::This, {"rhomjz"}, WhichSlice::RhomJzIons, {"rhomjz"});
    if (Hipace::m_deposit_rho) {
        add(lev, WhichSlice::This, {"rho"}, WhichSlice::RhomJzIons, {"rhomjz"});
    }
}

/** \brief Sets non zero Dirichlet Boundary conditions in RHS which is the source of the Poisson
 * equation: laplace LHS = RHS
 *
 * \param[in] RHS source of the Poisson equation: laplace LHS = RHS
 * \param[in] solver_size size of RHS/poisson solver (no tiling)
 * \param[in] geom geometry of of RHS/poisson solver
 * \param[in] offset shift boundary value by offset number of cells
 * \param[in] factor multiply the boundary_value by this factor
 * \param[in] boundary_value functional object (Real x, Real y) -> Real value_of_potential
 */
template<class Functional>
void
SetDirichletBoundaries (Array2<amrex::Real> RHS, const amrex::Box& solver_size,
                        const amrex::Geometry& geom, const amrex::Real offset,
                        const amrex::Real factor, const Functional& boundary_value)
{
    // To solve a Poisson equation with non-zero Dirichlet boundary conditions, the source term
    // must be corrected at the outmost grid points in x by -field_value_at_guard_cell / dx^2 and
    // in y by -field_value_at_guard_cell / dy^2, where dx and dy are those of the fine grid
    // This follows Van Loan, C. (1992). Computational frameworks for the fast Fourier transform.
    // Page 254 ff.
    // The interpolation is done in second order transversely and linearly in longitudinal direction
    const int box_len0 = solver_size.length(0);
    const int box_len1 = solver_size.length(1);
    const int box_lo0 = solver_size.smallEnd(0);
    const int box_lo1 = solver_size.smallEnd(1);
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);
    const amrex::Real offset0 = GetPosOffset(0, geom, solver_size);
    const amrex::Real offset1 = GetPosOffset(1, geom, solver_size);

    const amrex::Box edge_box = {{0, 0, 0}, {box_len0 + box_len1 - 1, 1, 0}};

    // ParallelFor only over the edge of the box
    amrex::ParallelFor(to2D(edge_box),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            const bool i_is_changing = (i < box_len0);
            const bool i_lo_edge = (!i_is_changing) && (!j);
            const bool i_hi_edge = (!i_is_changing) && j;
            const bool j_lo_edge = i_is_changing && (!j);
            const bool j_hi_edge = i_is_changing && j;

            const int i_idx = box_lo0 + i_hi_edge*(box_len0-1) + i_is_changing*i;
            const int j_idx = box_lo1 + j_hi_edge*(box_len1-1) + (!i_is_changing)*(i-box_len0);

            const amrex::Real i_idx_offset = i_idx + (- i_lo_edge + i_hi_edge) * offset;
            const amrex::Real j_idx_offset = j_idx + (- j_lo_edge + j_hi_edge) * offset;

            const amrex::Real x = i_idx_offset * dx + offset0;
            const amrex::Real y = j_idx_offset * dy + offset1;

            const amrex::Real dxdx = dx*dx*(!i_is_changing) + dy*dy*i_is_changing;

            // atomic add because the corners of RHS get two values
            amrex::Gpu::Atomic::AddNoRet(&(RHS(i_idx, j_idx)),
                                         - boundary_value(x, y) * factor / dxdx);
        });
}

void
Fields::SetBoundaryCondition (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                              const int which_slice, std::string component,
                              amrex::MultiFab&& staging_area,
                              amrex::Real offset, amrex::Real factor)
{
    const amrex::Box staging_box = geom[lev].Domain();

    if (lev == 0 && Hipace::m_boundary_field == FieldBoundary::Open) {
        HIPACE_PROFILE("Fields::SetOpenBoundaryCondition()");
        // Coarsest level: use Taylor expansion of the Green's function
        // to get Dirichlet boundary conditions

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(staging_area.size() == 1,
            "Open Boundaries only work for lev0 with everything in one box");
        amrex::FArrayBox& staging_area_fab = staging_area[0];

        const Array2<amrex::Real> arr_staging_area = staging_area_fab.array();

        const amrex::Real poff_x = GetPosOffset(0, geom[lev], staging_box);
        const amrex::Real poff_y = GetPosOffset(1, geom[lev], staging_box);
        const amrex::Real dx = geom[lev].CellSize(0);
        const amrex::Real dy = geom[lev].CellSize(1);
        // scale factor cancels out for all multipole coefficients except the 0th, for wich it adds
        // a constant term to the potential
        const amrex::Real scale = 3._rt/std::sqrt(
            pow<2>(geom[lev].ProbLength(0)) + pow<2>(geom[lev].ProbLength(1)));
        const amrex::Real radius = amrex::min(
            std::abs(geom[lev].ProbLo(0)), std::abs(geom[lev].ProbHi(0)),
            std::abs(geom[lev].ProbLo(1)), std::abs(geom[lev].ProbHi(1)));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(radius > 0._rt, "The x=0, y=0 coordinate must be inside"
            "the simulation box as it is used as the point of expansion for open boundaries");
        // ignore everything outside of 95% the min radius as the Taylor expansion only converges
        // outside of a circular patch containing the sources, i.e. the sources can't be further
        // from the center than the closest boundary as it would be the case in the corners
        const amrex::Real cutoff_sq = pow<2>(0.95_rt * radius * scale);
        const amrex::Real dxdy_div_4pi = dx*dy/(4._rt * MathConst::pi);

        MultipoleTuple coeff_tuple =
        amrex::ParReduce(MultipoleReduceOpList{}, MultipoleReduceTypeList{},
                         staging_area,
            [=] AMREX_GPU_DEVICE (int /*box_num*/, int i, int j, int) noexcept
            {
                const amrex::Real x = (i * dx + poff_x) * scale;
                const amrex::Real y = (j * dy + poff_y) * scale;
                if (x*x + y*y > cutoff_sq)  {
                    return amrex::IdentityTuple(MultipoleTuple{}, MultipoleReduceOpList{});
                }
                amrex::Real s_v = arr_staging_area(i, j);
                return GetMultipoleCoeffs(s_v, x, y);
            }
        );

        if (component == "Ez" || component == "Bz") {
            // Because Ez and Bz only have transverse derivatives of currents as sources, the
            // integral over the whole box is zero, meaning they have no physical monopole component
            amrex::get<0>(coeff_tuple) = 0._rt;
        }

        SetDirichletBoundaries(arr_staging_area, staging_box, geom[lev], offset, factor,
            [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y) noexcept
            {
                return dxdy_div_4pi*GetFieldMultipole(coeff_tuple, x*scale, y*scale);
            }
        );

    } else if (lev > 0) {
        HIPACE_PROFILE("Fields::SetMRBoundaryCondition()");
        // Fine level: interpolate solution from coarser level to get Dirichlet boundary conditions
        constexpr int interp_order = 2;

        auto solution_interp = interpolated_field_xy<interp_order, amrex::MultiFab>{
            getField(lev-1, which_slice, component), geom[lev-1]};

        for (amrex::MFIter mfi(staging_area, DfltMfi); mfi.isValid(); ++mfi)
        {
            const auto arr_solution_interp = solution_interp.array(mfi);
            const Array2<amrex::Real> arr_staging_area = staging_area.array(mfi);

            SetDirichletBoundaries(arr_staging_area, staging_box, geom[lev],
                                   offset, factor, arr_solution_interp);
        }
    }
}

void
Fields::LevelUpBoundary (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                         const int which_slice, const std::string& component,
                         const amrex::IntVect outer_edge, const amrex::IntVect inner_edge)
{
    if (lev == 0) return; // only interpolate boundaries to lev 1
    if (outer_edge == inner_edge) return;
    HIPACE_PROFILE("Fields::LevelUpBoundary()");
    constexpr int interp_order = 2;

    auto field_coarse_interp = interpolated_field_xy<interp_order, amrex::MultiFab>{
        getField(lev-1, which_slice, component), geom[lev-1]};
    amrex::MultiFab field_fine = getField(lev, which_slice, component);

    for (amrex::MFIter mfi( field_fine, DfltMfi); mfi.isValid(); ++mfi)
    {
        auto arr_field_coarse_interp = field_coarse_interp.array(mfi);
        const Array2<amrex::Real> arr_field_fine = field_fine.array(mfi);
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

        amrex::ParallelFor(to2D(fine_box_extended),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                // set interpolated values near edge of fine field between outer_edge and inner_edge
                // to compensate for incomplete charge/current deposition in those cells
                if(i<narrow_i_lo || i>narrow_i_hi || j<narrow_j_lo || j>narrow_j_hi) {
                    amrex::Real x = i * dx + offset0;
                    amrex::Real y = j * dy + offset1;
                    arr_field_fine(i,j) = arr_field_coarse_interp(x,y);
                }
            });
    }
}

void
Fields::LevelUp (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                 const int which_slice, const std::string& component)
{
    if (lev == 0) return; // only interpolate field to lev 1
    HIPACE_PROFILE("Fields::LevelUp()");
    constexpr int interp_order = 2;

    auto field_coarse_interp = interpolated_field_xy<interp_order, amrex::MultiFab>{
        getField(lev-1, which_slice, component), geom[lev-1]};
    amrex::MultiFab field_fine = getField(lev, which_slice, component);

    for (amrex::MFIter mfi( field_fine, DfltMfi); mfi.isValid(); ++mfi)
    {
        auto arr_field_coarse_interp = field_coarse_interp.array(mfi);
        const Array2<amrex::Real> arr_field_fine = field_fine.array(mfi);

        const amrex::Real dx = geom[lev].CellSize(0);
        const amrex::Real dy = geom[lev].CellSize(1);
        const amrex::Real offset0 = GetPosOffset(0, geom[lev], geom[lev].Domain());
        const amrex::Real offset1 = GetPosOffset(1, geom[lev], geom[lev].Domain());

        amrex::ParallelFor(to2D(field_fine[mfi].box()),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                // interpolate the full field
                const amrex::Real x = i * dx + offset0;
                const amrex::Real y = j * dy + offset1;
                arr_field_fine(i,j) = arr_field_coarse_interp(x,y);
            });
    }
}

void
Fields::SolvePoissonPsiExmByEypBxEzBz (amrex::Vector<amrex::Geometry> const& geom,
                                       const int current_N_level)
{
    /* Solves Laplacian(Psi) =  1/epsilon0 * -(rho-Jz/c) and
     * calculates Ex-c By, Ey + c Bx from  grad(-Psi)
     * Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
     * Solves Laplacian(Bz) = mu_0*(d_y(jx) - d_x(jy))
     */
    HIPACE_PROFILE("Fields::SolvePoissonPsiExmByEypBxEzBz()");

    PhysConst phys_const = get_phys_const();

    if (m_explicit && Hipace::m_do_beam_jz_minus_rho) {
        for (int lev=0; lev<current_N_level; ++lev) {
            add(lev, WhichSlice::This, {"rhomjz"}, WhichSlice::This, {"rhomjz_beam"});
        }
    }

    EnforcePeriodic(true, {Comps[WhichSlice::This]["jx"],
                           Comps[WhichSlice::This]["jy"],
                           Comps[WhichSlice::This]["rhomjz"]});
    for (int lev=0; lev<current_N_level; ++lev) {
        // interpolate rhomjz to lev from lev-1 in the domain edges
        LevelUpBoundary(geom, lev, WhichSlice::This, "rhomjz",
            amrex::IntVect{0, 0, 0}, -m_slices_nguards + amrex::IntVect{1, 1, 0});
        // interpolate jx and jy to lev from lev-1 in the domain edges and
        // also inside ghost cells to account for x and y derivative
        LevelUpBoundary(geom, lev, WhichSlice::This, "jx",
            amrex::IntVect{1, 1, 0}, -m_slices_nguards + amrex::IntVect{1, 1, 0});
        LevelUpBoundary(geom, lev, WhichSlice::This, "jy",
            amrex::IntVect{1, 1, 0}, -m_slices_nguards + amrex::IntVect{1, 1, 0});

        if (m_do_symmetrize) {
            SymmetrizeFields(Comps[WhichSlice::This]["rhomjz"], lev, 1, 1);
            SymmetrizeFields(Comps[WhichSlice::This]["jx"], lev, -1, 1);
            SymmetrizeFields(Comps[WhichSlice::This]["jy"], lev, 1, -1);
        }
    }

    for (int lev=0; lev<current_N_level; ++lev) {
        // Left-Hand Side for Poisson equation
        amrex::MultiFab lhs_Psi = getField(lev, WhichSlice::This, "Psi");
        amrex::MultiFab lhs_Ez  = getField(lev, WhichSlice::This, "Ez");
        amrex::MultiFab lhs_Bz  = getField(lev, WhichSlice::This, "Bz");

        // Psi: right-hand side 1/episilon0 * -(rho-Jz/c)
        Multiply(getStagingArea(lev),
            -1._rt/(phys_const.ep0), getField(lev, WhichSlice::This, "rhomjz"));

        SetBoundaryCondition(geom, lev, WhichSlice::This, "Psi", getStagingArea(lev),
            m_poisson_solver[lev]->BoundaryOffset(), m_poisson_solver[lev]->BoundaryFactor());

        m_poisson_solver[lev]->SolvePoissonEquation(lhs_Psi);

        // Ez: right-hand side 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
        LinCombination(getStagingArea(lev),
            1._rt/(phys_const.ep0*phys_const.c),
            derivative<Direction::x>{getField(lev, WhichSlice::This, "jx"), geom[lev]},
            1._rt/(phys_const.ep0*phys_const.c),
            derivative<Direction::y>{getField(lev, WhichSlice::This, "jy"), geom[lev]});

        SetBoundaryCondition(geom, lev, WhichSlice::This, "Ez", getStagingArea(lev),
            m_poisson_solver[lev]->BoundaryOffset(), m_poisson_solver[lev]->BoundaryFactor());

        m_poisson_solver[lev]->SolvePoissonEquation(lhs_Ez);

        // Bz: right-hand side mu_0*(d_y(jx) - d_x(jy))
        LinCombination(getStagingArea(lev),
            phys_const.mu0,
            derivative<Direction::y>{getField(lev, WhichSlice::This, "jx"), geom[lev]},
            -phys_const.mu0,
            derivative<Direction::x>{getField(lev, WhichSlice::This, "jy"), geom[lev]});

        SetBoundaryCondition(geom, lev, WhichSlice::This, "Bz", getStagingArea(lev),
            m_poisson_solver[lev]->BoundaryOffset(), m_poisson_solver[lev]->BoundaryFactor());

        m_poisson_solver[lev]->SolvePoissonEquation(lhs_Bz);
    }

    EnforcePeriodic(false, {Comps[WhichSlice::This]["Psi"],
                            Comps[WhichSlice::This]["Ez"],
                            Comps[WhichSlice::This]["Bz"]});

    for (int lev=0; lev<current_N_level; ++lev) {
        // interpolate fields to lev from lev-1 in the ghost cells
        LevelUpBoundary(geom, lev, WhichSlice::This, "Psi", m_slices_nguards, amrex::IntVect{0, 0, 0});
        LevelUpBoundary(geom, lev, WhichSlice::This, "Ez", m_slices_nguards, amrex::IntVect{0, 0, 0});
        LevelUpBoundary(geom, lev, WhichSlice::This, "Bz", m_slices_nguards, amrex::IntVect{0, 0, 0});
    }

    for (int lev=0; lev<current_N_level; ++lev) {
        // Compute ExmBy = -d/dx psi and EypBx = -d/dy psi
        amrex::MultiFab& slicemf = getSlices(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){
            const Array3<amrex::Real> arr = slicemf.array(mfi);
            const int Psi   = Comps[WhichSlice::This]["Psi"];
            const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
            const int EypBx = Comps[WhichSlice::This]["EypBx"];
            // number of ghost cells where ExmBy and EypBx are calculated is m_slices_nguards - 1
            const amrex::Box bx = mfi.growntilebox(m_slices_nguards - amrex::IntVect{1, 1, 0});
            const amrex::Real dx_inv = 0.5_rt*geom[lev].InvCellSize(Direction::x);
            const amrex::Real dy_inv = 0.5_rt*geom[lev].InvCellSize(Direction::y);

            amrex::ParallelFor(to2D(bx),
                [=] AMREX_GPU_DEVICE(int i, int j)
                {
                    // derivatives in x and y direction, no guards needed
                    arr(i,j,ExmBy) = - (arr(i+1,j,Psi) - arr(i-1,j,Psi))*dx_inv;
                    arr(i,j,EypBx) = - (arr(i,j+1,Psi) - arr(i,j-1,Psi))*dy_inv;
                });
        }
    }
}

void
Fields::SolvePoissonEz (amrex::Vector<amrex::Geometry> const& geom,
                        const int current_N_level, const int which_slice)
{
    /* Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonEz()");

    PhysConst phys_const = get_phys_const();

    EnforcePeriodic(true, {Comps[which_slice]["jx"],
                           Comps[which_slice]["jy"]});
    for (int lev=0; lev<current_N_level; ++lev) {
        // interpolate jx and jy to lev from lev-1 in the domain edges and
        // also inside ghost cells to account for x and y derivative
        LevelUpBoundary(geom, lev, which_slice, "jx",
            amrex::IntVect{1, 1, 0}, -m_slices_nguards + amrex::IntVect{1, 1, 0});
        LevelUpBoundary(geom, lev, which_slice, "jy",
            amrex::IntVect{1, 1, 0}, -m_slices_nguards + amrex::IntVect{1, 1, 0});

        if (m_do_symmetrize) {
            SymmetrizeFields(Comps[which_slice]["jx"], lev, -1, 1);
            SymmetrizeFields(Comps[which_slice]["jy"], lev, 1, -1);
        }
    }

    for (int lev=0; lev<current_N_level; ++lev) {
        // Left-Hand Side for Poisson equation
        amrex::MultiFab lhs_Ez = getField(lev, which_slice, "Ez");

        // Ez: right-hand side 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
        LinCombination(getStagingArea(lev),
            1._rt/(phys_const.ep0*phys_const.c),
            derivative<Direction::x>{getField(lev, which_slice, "jx"), geom[lev]},
            1._rt/(phys_const.ep0*phys_const.c),
            derivative<Direction::y>{getField(lev, which_slice, "jy"), geom[lev]});

        SetBoundaryCondition(geom, lev,which_slice, "Ez", getStagingArea(lev),
            m_poisson_solver[lev]->BoundaryOffset(), m_poisson_solver[lev]->BoundaryFactor());

        m_poisson_solver[lev]->SolvePoissonEquation(lhs_Ez);
    }

    EnforcePeriodic(false, {Comps[which_slice]["Ez"]});
    for (int lev=0; lev<current_N_level; ++lev) {
        // interpolate Ez to lev from lev-1 in the ghost cells
        LevelUpBoundary(geom, lev, which_slice, "Ez", m_slices_nguards, amrex::IntVect{0, 0, 0});
    }
}

void
Fields::SolvePoissonBxBy (amrex::Vector<amrex::Geometry> const& geom,
                          const int current_N_level, const int which_slice)
{
    /* Solves Laplacian(Bx) = mu_0*(- d_y(jz) + d_z(jy) )
     * Solves Laplacian(By) = mu_0*(d_x(jz) - d_z(jx) )
     * only used with predictor corrector solver
     */
    HIPACE_PROFILE("Fields::SolvePoissonBxBy()");

    PhysConst phys_const = get_phys_const();

    EnforcePeriodic(true, {Comps[WhichSlice::Next]["jx"],
                           Comps[WhichSlice::Next]["jy"],
                           Comps[WhichSlice::This]["jz"]});
    for (int lev=0; lev<current_N_level; ++lev) {
        // interpolate jx and jy to lev from lev-1 in the domain edges
        LevelUpBoundary(geom, lev, WhichSlice::Next, "jx", amrex::IntVect{0, 0, 0}, -m_slices_nguards);
        LevelUpBoundary(geom, lev, WhichSlice::Next, "jy", amrex::IntVect{0, 0, 0}, -m_slices_nguards);
        // interpolate jz to lev from lev-1 in the domain edges and
        // also inside ghost cells to account for x and y derivative
        LevelUpBoundary(geom, lev, WhichSlice::This, "jz", amrex::IntVect{1, 1, 0},
            -m_slices_nguards + amrex::IntVect{1, 1, 0});
        // jx and jy on WhichSlice::Previous was already leveled up on previous slice
        if (m_do_symmetrize) {
            SymmetrizeFields(Comps[WhichSlice::This]["jz"], lev, 1, 1);
            SymmetrizeFields(Comps[WhichSlice::Next]["jx"], lev, -1, 1);
            SymmetrizeFields(Comps[WhichSlice::Next]["jy"], lev, 1, -1);
        }
    }

    for (int lev=0; lev<current_N_level; ++lev) {
        // Left-Hand Side for Poisson equation
        amrex::MultiFab lhs_Bx = getField(lev, which_slice, "Bx");
        amrex::MultiFab lhs_By = getField(lev, which_slice, "By");

        // Bx: right-hand side mu_0*(- d_y(jz) + d_z(jy) )
        LinCombination(getStagingArea(lev),
                    -phys_const.mu0,
                    derivative<Direction::y>{getField(lev, WhichSlice::This, "jz"), geom[lev]},
                    phys_const.mu0,
                    derivative<Direction::z>{getField(lev, WhichSlice::Previous, "jy"),
                    getField(lev, WhichSlice::Next, "jy"), geom[lev]});

        SetBoundaryCondition(geom, lev, which_slice, "Bx", getStagingArea(lev),
            m_poisson_solver[lev]->BoundaryOffset(), m_poisson_solver[lev]->BoundaryFactor());

        m_poisson_solver[lev]->SolvePoissonEquation(lhs_Bx);

        // By: right-hand side mu_0*(d_x(jz) - d_z(jx) )
        LinCombination(getStagingArea(lev),
                   phys_const.mu0,
                   derivative<Direction::x>{getField(lev, WhichSlice::This, "jz"), geom[lev]},
                   -phys_const.mu0,
                   derivative<Direction::z>{getField(lev, WhichSlice::Previous, "jx"),
                   getField(lev, WhichSlice::Next, "jx"), geom[lev]});

        SetBoundaryCondition(geom, lev, which_slice, "By", getStagingArea(lev),
            m_poisson_solver[lev]->BoundaryOffset(), m_poisson_solver[lev]->BoundaryFactor());

        m_poisson_solver[lev]->SolvePoissonEquation(lhs_By);
    }

    EnforcePeriodic(false, {Comps[which_slice]["Bx"],
                            Comps[which_slice]["By"]});
    for (int lev=0; lev<current_N_level; ++lev) {
        // interpolate Bx and By to lev from lev-1 in the ghost cells
        LevelUpBoundary(geom, lev, which_slice, "Bx", m_slices_nguards, amrex::IntVect{0, 0, 0});
        LevelUpBoundary(geom, lev, which_slice, "By", m_slices_nguards, amrex::IntVect{0, 0, 0});
    }
}

void
Fields::SymmetrizeFields (int field_comp, const int lev, const int symm_x, const int symm_y)
{
    HIPACE_PROFILE("Fields::SymmetrizeFields()");

    AMREX_ALWAYS_ASSERT(symm_x*symm_x == 1 && symm_y*symm_y == 1);

    amrex::MultiFab& slicemf = getSlices(lev);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ) {
        const Array2<amrex::Real> arr = slicemf.array(mfi, field_comp);

        const amrex::Box full_box = mfi.growntilebox();

        const int upper_x = full_box.smallEnd(0) + full_box.bigEnd(0);
        const int upper_y = full_box.smallEnd(1) + full_box.bigEnd(1);

        amrex::Box quarter_box = full_box;
        quarter_box.setBig(0, full_box.smallEnd(0) + (full_box.length(0)+1)/2 - 1);
        quarter_box.setBig(1, full_box.smallEnd(1) + (full_box.length(1)+1)/2 - 1);

        amrex::ParallelFor(to2D(quarter_box),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                const amrex::Real avg = 0.25_rt*(arr(i, j) + arr(upper_x - i, j)*symm_x
                    + arr(i, upper_y - j)*symm_y + arr(upper_x - i, upper_y - j)*symm_x*symm_y);

                // Note: this may write to the same cell multiple times in the center.
                arr(i, j) = avg;
                arr(upper_x - i, j) = avg*symm_x;
                arr(i, upper_y - j) = avg*symm_y;
                arr(upper_x - i, upper_y - j) = avg*symm_x*symm_y;
            });
    }
}

void
Fields::EnforcePeriodic (const bool do_sum, std::vector<int>&& comp_idx)
{
    amrex::MultiFab& mfab = getSlices(0);

    if (!m_lev0_periodicity.isAnyPeriodic() && mfab.size() <= 1) {
        return; // no work to do
    }

    HIPACE_PROFILE("Fields::EnforcePeriodic()");

    // optimize adjacent fields to one FillBoundary call
    std::sort(comp_idx.begin(), comp_idx.end());
    int scomp = 0;
    int ncomp = 0;
    for (unsigned int i=0; i < comp_idx.size(); ++i) {
        if (ncomp==0) {
            scomp = comp_idx[i];
            ncomp = 1;
        }
        if (i+1 >= comp_idx.size() || comp_idx[i+1] > scomp+ncomp) {
            if (do_sum) {
                mfab.SumBoundary(scomp, ncomp, m_slices_nguards, m_lev0_periodicity);
            } else {
                mfab.FillBoundary(scomp, ncomp, m_slices_nguards, m_lev0_periodicity);
            }
            ncomp = 0;
        } else if (comp_idx[i+1] == scomp+ncomp) {
            ++ncomp;
        }
    }
}

void
Fields::InitialBfieldGuess (const amrex::Real relative_Bfield_error,
                            const amrex::Real predcorr_B_error_tolerance, const int lev)
{
    /* Sets the initial guess of the B field from the two previous slices
     */
    HIPACE_PROFILE("Fields::InitialBfieldGuess()");

    const amrex::Real mix_factor_init_guess = std::exp(-0.5_rt * std::pow(relative_Bfield_error /
                                              ( 2.5_rt * predcorr_B_error_tolerance ), 2));

    amrex::MultiFab& slicemf = getSlices(lev);

    AMREX_ALWAYS_ASSERT(Comps[WhichSlice::This]["Bx"]+1==Comps[WhichSlice::This]["By"]);
    AMREX_ALWAYS_ASSERT(Comps[WhichSlice::Previous]["Bx"]+1==Comps[WhichSlice::Previous]["By"]);
    AMREX_ALWAYS_ASSERT(Comps[WhichSlice::PCPrevIter]["Bx"]+1==Comps[WhichSlice::PCPrevIter]["By"]);

    amrex::MultiFab::LinComb(
        slicemf,
        1._rt+mix_factor_init_guess, slicemf, Comps[WhichSlice::Previous]["Bx"],
        -mix_factor_init_guess,      slicemf, Comps[WhichSlice::PCPrevIter]["Bx"],
        Comps[WhichSlice::This]["Bx"], 2, m_slices_nguards);
}

void
Fields::MixAndShiftBfields (const amrex::Real relative_Bfield_error,
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
    if (relative_Bfield_error != 0._rt || relative_Bfield_error_prev_iter != 0._rt)
    {
        weight_B_iter = relative_Bfield_error_prev_iter /
                        ( relative_Bfield_error + relative_Bfield_error_prev_iter );
        weight_B_prev_iter = relative_Bfield_error /
                             ( relative_Bfield_error + relative_Bfield_error_prev_iter );
    }
    else
    {
        weight_B_iter = 0.5_rt;
        weight_B_prev_iter = 0.5_rt;
    }

    amrex::MultiFab& slicemf = getSlices(lev);

    AMREX_ALWAYS_ASSERT(Comps[WhichSlice::This]["Bx"]+1==Comps[WhichSlice::This]["By"]);
    AMREX_ALWAYS_ASSERT(Comps[WhichSlice::PCIter]["Bx"]+1==Comps[WhichSlice::PCIter]["By"]);
    AMREX_ALWAYS_ASSERT(Comps[WhichSlice::PCPrevIter]["Bx"]+1==Comps[WhichSlice::PCPrevIter]["By"]);

    /* calculating the mixed temporary B field
     * B[WhichSlice::PCPrevIter] = c*B[WhichSlice::PCIter] + d*B[WhichSlice::PCPrevIter]. This is
     * temporarily stored in B[WhichSlice::PCPrevIter] just to avoid additional memory allocation.
     * B[WhichSlice::PCPrevIter] is overwritten at the end of this function */
    amrex::MultiFab::LinComb(
        slicemf,
        weight_B_iter,      slicemf, Comps[WhichSlice::PCIter    ]["Bx"],
        weight_B_prev_iter, slicemf, Comps[WhichSlice::PCPrevIter]["Bx"],
        Comps[WhichSlice::PCPrevIter]["Bx"], 2, m_slices_nguards);

    /* calculating the mixed B field  B = a*B + (1-a)*B_prev_iter */
    amrex::MultiFab::LinComb(
        slicemf,
        1._rt-predcorr_B_mixing_factor, slicemf, Comps[WhichSlice::This      ]["Bx"],
        predcorr_B_mixing_factor,       slicemf, Comps[WhichSlice::PCPrevIter]["Bx"],
        Comps[WhichSlice::This]["Bx"], 2, m_slices_nguards);

    /* Shifting the B field from the current iteration to the previous iteration */
    duplicate(lev, WhichSlice::PCPrevIter, {"Bx", "By"}, WhichSlice::PCIter, {"Bx", "By"});
}

amrex::Real
Fields::ComputeRelBFieldError (const int which_slice, const int which_slice_iter,
                               const amrex::Vector<amrex::Geometry>& geom,
                               const int current_N_level)
{
    // calculates the relative B field error between two B fields
    // for both Bx and By simultaneously
    HIPACE_PROFILE("Fields::ComputeRelBFieldError()");

    amrex::Real norm_Bdiff = 0._rt;
    amrex::Gpu::DeviceScalar<amrex::Real> gpu_norm_Bdiff(norm_Bdiff);
    amrex::Real* p_norm_Bdiff = gpu_norm_Bdiff.dataPtr();

    amrex::Real norm_B = 0._rt;
    amrex::Gpu::DeviceScalar<amrex::Real> gpu_norm_B(norm_B);
    amrex::Real* p_norm_B = gpu_norm_B.dataPtr();

    for (int lev=0; lev<current_N_level; ++lev) {

        amrex::MultiFab& slicemf = getSlices(lev);

        for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){
            const amrex::Box& bx = mfi.tilebox();

            Array3<amrex::Real const> const arr = slicemf.const_array(mfi);
            const int Bx_comp = Comps[which_slice]["Bx"];
            const int By_comp = Comps[which_slice]["By"];
            const int Bx_iter_comp = Comps[which_slice_iter]["Bx"];
            const int By_iter_comp = Comps[which_slice_iter]["By"];

            // factor to account for different cell size with MR
            const amrex::Real factor = geom[lev].CellSize(0) * geom[lev].CellSize(1) /
                (geom[0].CellSize(0) * geom[0].CellSize(1));

            amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), to2D(bx),
            [=] AMREX_GPU_DEVICE (int i, int j, amrex::Gpu::Handler const& handler) noexcept
            {
                amrex::Gpu::deviceReduceSum(p_norm_B, factor * std::sqrt(
                                            arr(i, j, Bx_comp) * arr(i, j, Bx_comp) +
                                            arr(i, j, By_comp) * arr(i, j, By_comp)),
                                            handler);
                amrex::Gpu::deviceReduceSum(p_norm_Bdiff, factor * std::sqrt(
                                ( arr(i, j, Bx_comp) - arr(i, j, Bx_iter_comp) ) *
                                ( arr(i, j, Bx_comp) - arr(i, j, Bx_iter_comp) ) +
                                ( arr(i, j, By_comp) - arr(i, j, By_iter_comp) ) *
                                ( arr(i, j, By_comp) - arr(i, j, By_iter_comp) )),
                                handler);
            }
            );
        }
    }
    norm_Bdiff = gpu_norm_Bdiff.dataValue();
    norm_B = gpu_norm_B.dataValue();

    // calculating the relative error
    const amrex::Real relative_Bfield_error = (norm_B > 0._rt) ? norm_Bdiff/norm_B : 0._rt;

    return relative_Bfield_error;
}

void
Fields::InSituComputeDiags (int step, amrex::Real time, int islice, const amrex::Geometry& geom3D,
                            int max_step, amrex::Real max_time)
{
    if (!utils::doDiagnostics(m_insitu_period, step, max_step, time, max_time)) return;
    HIPACE_PROFILE("Fields::InSituComputeDiags()");

    using namespace amrex::literals;

    constexpr int lev = 0;

    AMREX_ALWAYS_ASSERT(m_insitu_rdata.size()>0 && m_insitu_sum_rdata.size()>0 );

    const amrex::Real clight = get_phys_const().c;
    const amrex::Real dxdydz = geom3D.CellSize(0) * geom3D.CellSize(1) * geom3D.CellSize(2);
    const int nslices = geom3D.Domain().length(2);
    const int ExmBy = Comps[WhichSlice::This]["ExmBy"];
    const int EypBx = Comps[WhichSlice::This]["EypBx"];
    const int Ez = Comps[WhichSlice::This]["Ez"];
    const int Bx = Comps[WhichSlice::This]["Bx"];
    const int By = Comps[WhichSlice::This]["By"];
    const int Bz = Comps[WhichSlice::This]["Bz"];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Comps[WhichSlice::This].count("jz_beam") > 0,
        "Must use explicit solver for field insitu diagnostic");
    const int jz_beam = Comps[WhichSlice::This]["jz_beam"];

    amrex::MultiFab& slicemf = getSlices(lev);

    amrex::TypeMultiplier<amrex::ReduceOps, amrex::ReduceOpSum[m_insitu_nrp]> reduce_op;
    amrex::TypeMultiplier<amrex::ReduceData, amrex::Real[m_insitu_nrp]> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for ( amrex::MFIter mfi(slicemf, DfltMfi); mfi.isValid(); ++mfi ) {
        Array3<amrex::Real const> const arr = slicemf.const_array(mfi);
        reduce_op.eval(
            mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int) -> ReduceTuple
            {
                return {                                            // Tuple contains:
                    pow<2>(arr(i,j,ExmBy) + arr(i,j,By) * clight),  // 0    [Ex^2]
                    pow<2>(arr(i,j,EypBx) - arr(i,j,Bx) * clight),  // 1    [Ey^2]
                    pow<2>(arr(i,j,Ez)),                            // 2    [Ez^2]
                    pow<2>(arr(i,j,Bx)),                            // 3    [Bx^2]
                    pow<2>(arr(i,j,By)),                            // 4    [By^2]
                    pow<2>(arr(i,j,Bz)),                            // 5    [Bz^2]
                    pow<2>(arr(i,j,ExmBy)),                         // 6    [ExmBy^2]
                    pow<2>(arr(i,j,EypBx)),                         // 7    [EypBx^2]
                    arr(i,j,jz_beam),                               // 8    [jz_beam]
                    arr(i,j,Ez)*arr(i,j,jz_beam)                    // 9    [Ez*jz_beam]
                };
            });
    }

    auto real_arr = amrex::tupleToArray(reduce_data.value());

    for (int i=0; i<m_insitu_nrp; ++i) {
        m_insitu_rdata[islice + i * nslices] = real_arr[i] * dxdydz;
        m_insitu_sum_rdata[i] += real_arr[i] * dxdydz;
    }
}

void
Fields::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom3D,
                           int max_step, amrex::Real max_time)
{
    if (!utils::doDiagnostics(m_insitu_period, step, max_step, time, max_time)) return;
    HIPACE_PROFILE("Fields::InSituWriteToFile()");

#ifdef HIPACE_USE_OPENPMD
    // create subdirectory
    openPMD::auxiliary::create_directories(m_insitu_file_prefix);
#endif

    // zero pad the rank number;
    std::string::size_type n_zeros = 4;
    std::string rank_num = std::to_string(amrex::ParallelDescriptor::MyProc());
    std::string pad_rank_num = std::string(n_zeros-std::min(rank_num.size(), n_zeros),'0')+rank_num;

    // open file
    std::ofstream ofs{m_insitu_file_prefix + "/reduced_fields." + pad_rank_num + ".txt",
        std::ofstream::out | std::ofstream::app | std::ofstream::binary};

    const int nslices_int = geom3D.Domain().length(2);
    const std::size_t nslices = static_cast<std::size_t>(nslices_int);
    const int is_normalized_units = Hipace::m_normalized_units;

    // specify the structure of the data later available in python
    // avoid pointers to temporary objects as second argument, stack variables are ok
    const amrex::Vector<insitu_utils::DataNode> all_data{
        {"time"     , &time},
        {"step"     , &step},
        {"n_slices" , &nslices_int},
        {"z_lo"     , &geom3D.ProbLo()[2]},
        {"z_hi"     , &geom3D.ProbHi()[2]},
        {"is_normalized_units", &is_normalized_units},
        {"[Ex^2]"   , &m_insitu_rdata[0], nslices},
        {"[Ey^2]"   , &m_insitu_rdata[1*nslices], nslices},
        {"[Ez^2]"   , &m_insitu_rdata[2*nslices], nslices},
        {"[Bx^2]"   , &m_insitu_rdata[3*nslices], nslices},
        {"[By^2]"   , &m_insitu_rdata[4*nslices], nslices},
        {"[Bz^2]"   , &m_insitu_rdata[5*nslices], nslices},
        {"[ExmBy^2]", &m_insitu_rdata[6*nslices], nslices},
        {"[EypBx^2]", &m_insitu_rdata[7*nslices], nslices},
        {"[jz_beam]", &m_insitu_rdata[8*nslices], nslices},
        {"[Ez*jz_beam]", &m_insitu_rdata[9*nslices], nslices},
        {"integrated", {
            {"[Ex^2]"   , &m_insitu_sum_rdata[0]},
            {"[Ey^2]"   , &m_insitu_sum_rdata[1]},
            {"[Ez^2]"   , &m_insitu_sum_rdata[2]},
            {"[Bx^2]"   , &m_insitu_sum_rdata[3]},
            {"[By^2]"   , &m_insitu_sum_rdata[4]},
            {"[Bz^2]"   , &m_insitu_sum_rdata[5]},
            {"[ExmBy^2]", &m_insitu_sum_rdata[6]},
            {"[EypBx^2]", &m_insitu_sum_rdata[7]},
            {"[jz_beam]", &m_insitu_sum_rdata[8]},
            {"[Ez*jz_beam]", &m_insitu_sum_rdata[9]}
        }}
    };

    if (ofs.tellp() == 0) {
        // write JSON header containing a NumPy structured datatype
        insitu_utils::write_header(all_data, ofs);
    }

    // write binary data according to datatype in header
    insitu_utils::write_data(all_data, ofs);

    // close file
    ofs.close();
    // assert no file errors
#ifdef HIPACE_USE_OPENPMD
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu field diagnostics");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu field diagnostics. "
        "Maybe the specified subdirectory does not exist");
#endif

    // reset arrays for insitu data
    for (auto& x : m_insitu_rdata) x = 0.;
    for (auto& x : m_insitu_sum_rdata) x = 0.;
}
