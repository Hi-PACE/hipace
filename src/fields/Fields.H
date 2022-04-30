/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Remi Lehe
 * Severin Diederichs, WeiqunZhang, coulibaly-mouhamed
 * License: BSD-3-Clause-LBNL
 */
#ifndef FIELDS_H_
#define FIELDS_H_

#include "fft_poisson_solver/FFTPoissonSolver.H"
#include "diagnostics/Diagnostic.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

#include <algorithm>

class Hipace;

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichSlice {
    enum slice { Next=0, This, Previous1, Previous2, RhoIons, N };
};

struct assert_map : std::map<std::string, int> {
    int operator[] (const std::string& str) const {
        if (count(str)==0) {
            amrex::ErrorStream() << "Component '"+str+"' is not allocated\nFields allocated:";
            for (auto& [alloc_str, num] : *this) {
                amrex::ErrorStream() << " '" << alloc_str << "' (" << num << "),";
            }
            amrex::ErrorStream() << "\n";
            AMREX_ALWAYS_ASSERT(count(str)!=0);
        }
        return at(str);
    }

    template<class...Args>
    void multi_emplace(int& n, Args...comps) {
        (emplace(comps, n++),...);
    }
};

/** \brief Map names and indices of each fields in each slice
 */
inline std::array<assert_map, 5> Comps{};

/** \brief number of fields in each slice
 */
inline std::array<int, 5> N_Comps {0,0,0,0,0};

/** \brief Direction of each dimension. Can be used for clean handling 2D vs. 3D in the future */
struct Direction{
    enum dir{x=0, y, z};
};

/** \brief Function to help converting field indexes to positions and backwards. Usage:
 * x = i * dx + GetPosOffset(0, geom, box)
 * i = round( (x - GetPosOffset(0, geom, box)) / dx )
 *
 * \param[in] dir 0, 1 or 2 for x, y or z respectively
 * \param[in] geom geometry of the field
 * \param[in] box box of the field, can be with or without ghost cells
 */
inline amrex::Real
GetPosOffset (const int dir, const amrex::Geometry& geom, const amrex::Box& box) {
    using namespace amrex::literals;
    // match boxes at center point
    return 0.5_rt * (geom.ProbLo(dir) + geom.ProbHi(dir)
           - geom.CellSize(dir) * (box.smallEnd(dir) + box.bigEnd(dir)));
}

/** \brief Main class handling all field data structures and operations
 *
 * This is a fundamental class of Hipace, handling initialization of the fields,
 * operations on arrays, exchanges between the 3D array and 2D slices.
 */
class Fields
{
private:
    /** Number of slices in temporary slice object */
    static constexpr int m_nslices = 5;

public:
    /** Constructor */
    explicit Fields (Hipace const* a_hipace);

    /** Allocate MultiFabs for the 3D array and the 2D slices
     * and define the BoxArrays and DistributionMappings.
     * \param[in] lev MR level
     * \param[in] geom Geometry
     * \param[in] slice_ba BoxArray for the slice
     * \param[in] slice_dm DistributionMapping for the slice
     * \param[in] bin_size size of plasma tiles, for tmp arrays
     */
    void AllocData (
        int lev, amrex::Vector<amrex::Geometry> const& geom, const amrex::BoxArray& slice_ba,
        const amrex::DistributionMapping& slice_dm, int bin_size);

    /** Class to handle transverse FFT Poisson solver on 1 slice */
    amrex::Vector<std::unique_ptr<FFTPoissonSolver>> m_poisson_solver;
    /** get function for the 2D slices */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>>& getSlices () {return m_slices; }
    /** get function for the 2D slices
     * \param[in] lev MR level
     */
    std::array<amrex::MultiFab, m_nslices>& getSlices (int lev) {return m_slices[lev]; }
    /** get function for the 2D slices
     * \param[in] lev MR level
     * \param[in] islice slice index
     */
    amrex::MultiFab& getSlices (int lev, int islice) {return m_slices[lev][islice]; }
    /** get function for the 2D slices (const version)
     * \param[in] lev MR level
     * \param[in] islice slice index
     */
    const amrex::MultiFab& getSlices (int lev, int islice) const {return m_slices[lev][islice]; }
    /** get amrex::MultiFab of a field in a slice
     * \param[in] lev MR level
     * \param[in] islice slice index
     * \param[in] comp component name of field (see Comps)
     */
    amrex::MultiFab getField (const int lev, const int islice, const std::string comp) {
        return amrex::MultiFab(getSlices(lev, islice), amrex::make_alias, Comps[islice][comp], 1);
    }
    /** get amrex::MultiFab of the poisson staging area
     * \param[in] lev MR level
     */
    amrex::MultiFab getStagingArea (const int lev) {
        return amrex::MultiFab(m_poisson_solver[lev]->StagingArea(), amrex::make_alias, 0, 1);
    }
    /** Return reference to density tile arrays */
    amrex::Vector<amrex::FArrayBox>& getTmpDensities() { return m_tmp_densities; }
    /** check whether the fields are initialized correctly */
    void checkInit() {
        for (auto& slices_lev : m_slices) {
            for (auto& slice : slices_lev) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(slice.nGrowVect() == m_slices_nguards,
                    "m_slices[lev][islice].nGrowVect() must be equal to m_slices_nguards");
            }
        }
    }
    /** \brief Copy between the full FArrayBox and slice MultiFab.
     *
     * \param[in] lev MR level
     * \param[in] i_slice z slice in which to write the data
     * \param[in] diag_geom diagnostics geometry
     * \param[in,out] diag_fab full FArrayBox of diagnostics
     * \param[in] diag_box box on which diag_fab is defined
     * \param[in] calc_geom main geometry
     * \param[in] diag_comps_vect the field components to copy
     * \param[in] ncomp number of components to copy
     */
     void Copy (const int lev, const int i_slice, const amrex::Geometry& diag_geom,
                amrex::FArrayBox& diag_fab, amrex::Box diag_box, const amrex::Geometry& calc_geom,
                const amrex::Gpu::DeviceVector<int>& diag_comps_vect, const int ncomp);

    /** \brief Shift slices by 1 element: slices (1,2) are then stored in (2,3).
     *
     * When looping over slices from head to tail, the same slice MultiFabs are used
     * to compute each slice. The current slice is always stored in index 1.
     * Hence, after one slice is computed, slices must be shifted by 1 element.
     *
     * \param[in] nlev number of MR level
     * \param[in] islice current slice to be shifted
     * \param[in] geom geometry
     * \param[in] patch_lo lower longitudinal end of refined level
     * \param[in] patch_hi upper longitudinal end of refined level
     */
    void ShiftSlices (int nlev, int islice, const amrex::Geometry geom, const amrex::Real patch_lo,
                      const amrex::Real patch_hi);

    /** add rho of the ions to rho (this slice) */
    void AddRhoIons (const int lev, bool inverse=false);

    /** \brief Set up boundary conditions before poisson solve
     * lev==0: leave at zero
     * lev!=0: interpolate boundaries from lev-1
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] component which can be Psi, Ez, By, Bx ...
     * \param[in] islice longitudinal slice
     */
    void SetBoundaryCondition (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                               std::string component, const int islice);

    /** \brief Interpolate values from coarse grid to the fine grid
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] component which can be Psi or rho
     * \param[in] islice longitudinal slice
     * \param[in] outer_edge start writing interpolated values at domain + outer_edge
     * \param[in] inner_edge stop writing interpolated values at domain + inner_edge
     */
    void InterpolateFromLev0toLev1 (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                                    std::string component, const int islice,
                                    const amrex::IntVect outer_edge,
                                    const amrex::IntVect inner_edge);

    /** \brief Compute ExmBy and EypBx on the slice container from J by solving a Poisson equation
     * ExmBy and EypBx are solved in the same function because both rely on Psi.
     *
     * \param[in] geom Geometry
     * \param[in] m_comm_xy transverse communicator on the slice
     * \param[in] lev current level
     * \param[in] islice longitudinal slice
     */
    void SolvePoissonExmByAndEypBx (amrex::Vector<amrex::Geometry> const& geom,
                                    const MPI_Comm& m_comm_xy, const int lev, const int islice);
    /** \brief Compute Ez on the slice container from J by solving a Poisson equation
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] islice longitudinal slice
     */
    void SolvePoissonEz (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                         const int islice);
    /** \brief Compute Bx on the slice container from J by solving a Poisson equation
     *
     * \param[in,out] Bx_iter Bx field during current iteration of the predictor-corrector loop
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] islice longitudinal slice
     */
    void SolvePoissonBx (amrex::MultiFab& Bx_iter, amrex::Vector<amrex::Geometry> const& geom,
                         const int lev, const int islice);
    /** \brief Compute By on the slice container from J by solving a Poisson equation
     *
     * \param[in,out] By_iter By field during current iteration of the predictor-corrector loop
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] islice longitudinal slice
     */
    void SolvePoissonBy (amrex::MultiFab& By_iter, amrex::Vector<amrex::Geometry> const& geom,
                         const int lev, const int islice);
    /** \brief Compute Bz on the slice container from J by solving a Poisson equation
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] islice longitudinal slice
     */
    void SolvePoissonBz (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                         const int islice);
    /** \brief Sets the initial guess of the B field from the two previous slices
     *
     * This modifies component Bx or By of slice 1 in m_fields.m_slices
     *
     * \param[in] relative_Bfield_error relative B field error used to determine the mixing factor
     * \param[in] predcorr_B_error_tolerance relative B field error tolerance in the predcorr loop
     * \param[in] lev current level
     */
    void InitialBfieldGuess (const amrex::Real relative_Bfield_error,
                             const amrex::Real predcorr_B_error_tolerance, const int lev);
    /** \brief Mixes the B field with the calculated current and previous iteration
     * of it and shifts the current to the previous iteration afterwards.
     * This modifies component Bx or By of slice 1 in m_fields.m_slices
     *
     * \param[in] B_iter B field during current iteration of the predictor-corrector loop
     * \param[in,out] B_prev_iter B field during previous iteration of the pred.-cor. loop
     * \param[in] field_comp field component to be mixed (usually Bx or By)
     * \param[in] relative_Bfield_error relative B field error used to determine the mixing factor
     * \param[in] relative_Bfield_error_prev_iter relative B field error of the previous iteration
     * \param[in] predcorr_B_mixing_factor mixing factor for B fields in predcorr loop
     * \param[in] lev current level
     */
    void MixAndShiftBfields (const amrex::MultiFab& B_iter, amrex::MultiFab& B_prev_iter,
                             const int field_comp, const amrex::Real relative_Bfield_error,
                             const amrex::Real relative_Bfield_error_prev_iter,
                             const amrex::Real predcorr_B_mixing_factor, const int lev);

    /** \brief Function to calculate the relative B field error
     * used in the predictor corrector loop
     *
     * \param[in] Bx Bx field (input as MutliFab)
     * \param[in] By By field (input as MutliFab)
     * \param[in] Bx_iter Bx field from the previous iteration (input as MutliFab)
     * \param[in] By_iter By field from the previous iteration (input as MutliFab)
     * \param[in] Bx_comp component of the Bx field in the MultiFab
     *            (usually either Bx or 0)
     * \param[in] By_comp component of the By field in the MultiFab
     *            (usually either By or 0)
     * \param[in] Bx_iter_comp component of the Bx field of the previous iteration in the MultiFab
     *            (usually either Bx or 0)
     * \param[in] By_iter_comp component of the By field of the previous iteration in the MultiFab
     *            (usually either By or 0)
     * \param[in] geom Geometry of the problem
     */
    amrex::Real ComputeRelBFieldError (const amrex::MultiFab& Bx, const amrex::MultiFab& By,
                                       const amrex::MultiFab& Bx_iter,
                                       const amrex::MultiFab& By_iter, const int Bx_comp,
                                       const int By_comp, const int Bx_iter_comp,
                                       const int By_iter_comp, const amrex::Geometry& geom);

    /** \brief set all selected fields to a value
     *
     * \param[in] val value
     * \param[in] lev level of fields
     * \param[in] islice slice of fields
     * \param[in] comps names of fields like "Ez", "Bx", "By", "Bz"
     */
    template<class...Args>
    void setVal (const amrex::Real val, const int lev, const int islice, Args...comps) {
        static constexpr int ncomps = sizeof...(comps);
        amrex::GpuArray<int, ncomps> c_idx = {Comps[islice][comps]...};
        amrex::MultiFab& mfab = getSlices(lev, islice);

        for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
            amrex::Array4<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(), ncomps,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    array(i,j,k,c_idx[n]) = val;
                });
        }
    }

    /** \brief multiply all selected fields with a value
     *
     * \param[in] val value
     * \param[in] lev level of fields
     * \param[in] islice slice of fields
     * \param[in] comps names of fields like "Ez", "Bx", "By", "Bz"
     */
    template<class...Args>
    void mult (const amrex::Real val, const int lev, const int islice, Args...comps) {
        static constexpr int ncomps = sizeof...(comps);
        amrex::GpuArray<int, ncomps> c_idx = {Comps[islice][comps]...};
        amrex::MultiFab& mfab = getSlices(lev, islice);

        for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
            amrex::Array4<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(), ncomps,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    array(i,j,k,c_idx[n]) *= val;
                });
        }
    }

    /** \brief copy all selected fields between slices
     *
     * \param[in] lev level of fields
     * \param[in] islice_dst destination slice
     * \param[in] islice_src source slice
     * \param[in] comps names of fields like "Ez", "Bx", "By", "Bz"
     */
    template<class...Args>
    void shift (const int lev, const int islice_dst, const int islice_src, Args...comps) {
        static constexpr int ncomps = sizeof...(comps);
        amrex::GpuArray<int, ncomps> c_idx_src = {Comps[islice_src][comps]...};
        amrex::GpuArray<int, ncomps> c_idx_dst = {Comps[islice_dst][comps]...};
        amrex::MultiFab& mfab_src = getSlices(lev, islice_src);
        amrex::MultiFab& mfab_dst = getSlices(lev, islice_dst);

        for (amrex::MFIter mfi(mfab_dst); mfi.isValid(); ++mfi) {
            amrex::Array4<amrex::Real> array_src = mfab_src.array(mfi);
            amrex::Array4<amrex::Real> array_dst = mfab_dst.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(), ncomps,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    array_dst(i,j,k,c_idx_dst[n]) = array_src(i,j,k,c_idx_src[n]);
                });
        }
    }

    /** \brief call amrex FillBoundary for multiple fields
     *
     * \param[in] period periodicity of the fields
     * \param[in] lev level of fields
     * \param[in] islice slice of fields
     * \param[in] comps names of fields like "Ez", "Bx", "By", "Bz"
     */
    template<class...Args>
    void FillBoundary (const amrex::Periodicity& period, const int lev, const int islice,
                       Args...comps) {
        std::array<int, sizeof...(comps)> c_idx = {Comps[islice][comps]...};
        amrex::MultiFab& mfab = getSlices(lev, islice);

        // optimize adjacent fields to one FillBoundary call
        std::sort(c_idx.begin(), c_idx.end());
        int scomp = 0;
        int ncomp = 0;
        for (unsigned int i=0; i < c_idx.size(); ++i) {
            if (ncomp==0) {
                scomp = c_idx[i];
                ncomp = 1;
            }
            if (i+1 >= c_idx.size() || c_idx[i+1] > scomp+ncomp) {
                mfab.FillBoundary(scomp, ncomp, period);
                ncomp = 0;
            } else if (c_idx[i+1] == scomp+ncomp) {
                ++ncomp;
            }
        }
    }

    /** Number of guard cells for slices MultiFab */
    static amrex::IntVect m_slices_nguards;
    /** Number of guard cells for poisson solver MultiFab */
    static amrex::IntVect m_poisson_nguards;
    /** If the poisson solver should include the guard cells */
    bool m_extended_solve = false;

private:
    /** Vector over levels, array of 4 slices required to compute current slice */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>> m_slices;
    /** Whether to use Dirichlet BC for the Poisson solver. Otherwise, periodic */
    bool m_do_dirichlet_poisson = true;
    /** Temporary density arrays. one per OpenMP thread, used when tiling is on. */
    amrex::Vector<amrex::FArrayBox> m_tmp_densities;
    /** Postfix name of all charges and currents on This slice in the current simulation */
    amrex::Vector<std::string> m_all_charge_currents_names;
    /** Stores temporary values for z interpolation in Fields::Copy */
    amrex::Gpu::DeviceVector<amrex::Real> m_rel_z_vec;
    /** Number of guard cells where ExmBy and EypBx are calculated */
    amrex::IntVect m_exmby_eypbx_nguard;
    /** Number of guard cells where sources for poisson equation are included */
    amrex::IntVect m_source_nguard;
    /** If lev_0 should be solved with open boundary conditions */
    bool m_open_boundary = false;
    /** If the explicit solver is being used */
    bool m_explicit = false;
    /** If any plasma species has a neutralizing background */
    bool m_any_neutral_background = false;
};

#endif
