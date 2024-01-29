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
#include "laser/MultiLaser.H"
#include "utils/GPUUtil.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

#include <algorithm>

class Hipace;
class MultiLaser;

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichSlice {
    enum slice { Next=0, This, Previous, RhomJzIons, Salame, PCIter, PCPrevIter, N };
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
inline std::array<assert_map, WhichSlice::N> Comps{};

/** \brief number of fields in each slice
 */
inline int N_Comps {0};

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
public:
    /** Constructor */
    explicit Fields (const int nlev);

    /** Allocate MultiFabs for the 3D array and the 2D slices
     * and define the BoxArrays and DistributionMappings.
     * \param[in] lev MR level
     * \param[in] geom Geometry
     * \param[in] slice_ba BoxArray for the slice
     * \param[in] slice_dm DistributionMapping for the slice
     * \param[in] bin_size size of plasma tiles, for tmp arrays
     */
    void AllocData (
        int lev, amrex::Geometry const& geom, const amrex::BoxArray& slice_ba,
        const amrex::DistributionMapping& slice_dm, int bin_size);

    /** get function for the 2D slices */
    amrex::Vector<amrex::MultiFab>& getSlices () {return m_slices; }
    /** get function for the 2D slices
     * \param[in] lev MR level
     */
    amrex::MultiFab& getSlices (int lev) {return m_slices[lev]; }
    /** get function for the 2D slices (const version)
     * \param[in] lev MR level
     */
    const amrex::MultiFab& getSlices (int lev) const {return m_slices[lev]; }
    /** get amrex::MultiFab of a field in a slice
     * \param[in] lev MR level
     * \param[in] islice slice index
     * \param[in] comp component name of field (see Comps)
     */
    amrex::MultiFab getField (const int lev, const int islice, const std::string comp) {
        return amrex::MultiFab(getSlices(lev), amrex::make_alias, Comps[islice][comp], 1);
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
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                slices_lev.nComp() == 0 || slices_lev.nGrowVect() == m_slices_nguards,
                "m_slices[lev].nGrowVect() must be equal to m_slices_nguards"
                "for initialized slices");
        }
    }
    /** \brief Copy between the full FArrayBox and slice MultiFab.
     *
     * \param[in] lev MR level
     * \param[in] i_slice z slice in which to write the data
     * \param[in,out] fd data for field diagnostics
     * \param[in] calc_geom main geometry
     * \param[in] multi_laser MultiLaser object
     */
     void Copy (const int lev, const int i_slice, FieldDiagnosticData& fd,
                const amrex::Geometry& calc_geom, MultiLaser& multi_laser);

     /** \brief Initialize all required fields to zero and interpolate from lev-1 to lev if needed
     *
     * \param[in] lev the MR level
     * \param[in] islice current slice index
     * \param[in] geom 3D Geometry
     */
    void InitializeSlices (int lev, int islice, const amrex::Vector<amrex::Geometry>& geom);

    /** \brief Shift slices by 1 element: slices (1,2) are then stored in (2,3).
     *
     * When looping over slices from head to tail, the same slice MultiFabs are used
     * to compute each slice. The current slice is always stored in index 1.
     * Hence, after one slice is computed, slices must be shifted by 1 element.
     *
     * \param[in] lev the MR level
     */
    void ShiftSlices (int lev);

    /** add rho of the ions to rho (this slice) */
    void AddRhoIons (const int lev);

    /** \brief Set up boundary conditions before poisson solve
     * lev==0: leave at zero or add open boundaries
     * lev>0: interpolate boundaries from lev-1
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] which_slice slice of the field
     * \param[in] component which can be Psi, Ez, By, Bx ...
     * \param[in,out] staging_area Target MultiFab where the boundary condition is applied
     * \param[in] box_grow relative to the domain, how large is the box of staging_area
     * \param[in] offset shift boundary value by offset number of cells
     * \param[in] factor multiply the boundary value by this factor
     */
    void SetBoundaryCondition (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                               const int which_slice, std::string component,
                               amrex::MultiFab&& staging_area, amrex::IntVect box_grow,
                               amrex::Real offset, amrex::Real factor);

    /** \brief Interpolate values from coarse grid (lev-1) to the boundary of the fine grid (lev).
     * This may include ghost cells.
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] which_slice slice of the field to interpolate
     * \param[in] component which can be Psi or rho etc.
     * \param[in] outer_edge start writing interpolated values at domain + outer_edge
     * \param[in] inner_edge stop writing interpolated values at domain + inner_edge
     */
    void LevelUpBoundary (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                          const int which_slice, const std::string& component,
                          const amrex::IntVect outer_edge, const amrex::IntVect inner_edge);

    /** \brief Interpolate the full field from the coarse grid (lev-1) to the fine grid (lev).
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     * \param[in] which_slice slice of the field to interpolate
     * \param[in] component which can be jx_beam or jy_beam etc.
     */
    void LevelUp (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                  const int which_slice, const std::string& component);

    /** \brief Compute Psi, ExmBy, EypBx, Ez and Bz on the slice container from J by solving three
     * poisson equations. ExmBy and EypBx computed from grad(-Psi).
     * This function does all the necessary boundary interpolation between MR levels
     *
     * \param[in] geom Geometry
     * \param[in] current_N_level number of MR levels active on the current slice
     */
    void SolvePoissonPsiExmByEypBxEzBz (amrex::Vector<amrex::Geometry> const& geom,
                                        const int current_N_level);
    /** \brief Compute Ez on the slice container from J by solving a Poisson equation.
     * This function does all the necessary boundary interpolation between MR levels
     *
     * \param[in] geom Geometry
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] which_slice defines if this or the salame slice is handled
     */
    void SolvePoissonEz (amrex::Vector<amrex::Geometry> const& geom, const int current_N_level,
                         const int which_slice = WhichSlice::This);
    /** \brief Compute Bx and By on the slice container from J by solving two Poisson equations.
     * This function does all the necessary boundary interpolation between MR levels
     *
     * \param[in] geom Geometry
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] which_slice slice to put the result into
     */
    void SolvePoissonBxBy (amrex::Vector<amrex::Geometry> const& geom, const int current_N_level,
                           const int which_slice);
    /** \brief Symmetrize fields by averaging over (x,y), symm_x*(-x,y),
     * symm_y*(x,-y) and symm_x*symm_y*(-x,-y) where symm_x and symm_y can be 1 or -1.
     *
     * \param[in] field_comp component index of the filed
     * \param[in] lev current level
     * \param[in] symm_x type of reflection in x direction
     * \param[in] symm_y type of reflection in y direction
     */
    void SymmetrizeFields (int field_comp, const int lev, const int symm_x, const int symm_y);
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
     * \param[in] relative_Bfield_error relative B field error used to determine the mixing factor
     * \param[in] relative_Bfield_error_prev_iter relative B field error of the previous iteration
     * \param[in] predcorr_B_mixing_factor mixing factor for B fields in predcorr loop
     * \param[in] lev current level
     */
    void MixAndShiftBfields (const amrex::Real relative_Bfield_error,
                             const amrex::Real relative_Bfield_error_prev_iter,
                             const amrex::Real predcorr_B_mixing_factor, const int lev);

    /** \brief Function to calculate the relative B field error
     * used in the predictor corrector loop
     *
     * \param[in] which_slice slice of Bx and By field
     * \param[in] which_slice_iter slice of Bx and By field from the previous iteration
     * \param[in] geom Geometry of the problem
     * \param[in] current_N_level number of MR levels active on the current slice
     */
    amrex::Real ComputeRelBFieldError (const int which_slice, const int which_slice_iter,
        const amrex::Vector<amrex::Geometry>& geom, const int current_N_level);


    /** Compute in-situ field diagnostics of current slice, store in member variable
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] geom3D Geometry of the problem
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituComputeDiags (int step, amrex::Real time, int islice, const amrex::Geometry& geom3D,
                             int max_step, amrex::Real max_time);

    /** Dump in-situ reduced diagnostics to file.
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] geom3D Geometry object for the whole domain
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom3D,
                            int max_step, amrex::Real max_time);

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
        const amrex::GpuArray<int, ncomps> c_idx = {Comps[islice][comps]...};
        amrex::MultiFab& mfab = getSlices(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mfab, DfltMfiTlng); mfi.isValid(); ++mfi) {
            const Array3<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int)
                {
                    for (int n=0; n<ncomps; ++n) {
                        array(i,j,c_idx[n]) = val;
                    }
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
        const amrex::GpuArray<int, ncomps> c_idx = {Comps[islice][comps]...};
        amrex::MultiFab& mfab = getSlices(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mfab, DfltMfiTlng); mfi.isValid(); ++mfi) {
            const Array3<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int)
                {
                    for (int n=0; n<ncomps; ++n) {
                        array(i,j,c_idx[n]) *= val;
                    }
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
        const amrex::GpuArray<int, ncomps> c_idx_src = {Comps[islice_src][comps]...};
        const amrex::GpuArray<int, ncomps> c_idx_dst = {Comps[islice_dst][comps]...};
        amrex::MultiFab& mfab = getSlices(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mfab, DfltMfiTlng); mfi.isValid(); ++mfi) {
            const Array3<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int)
                {
                    for (int n=0; n<ncomps; ++n) {
                        array(i,j,c_idx_dst[n]) = array(i,j,c_idx_src[n]);
                    }
                });
        }
    }

    /** \brief copy all selected fields between slices or on the same slice
     * Uses references to C-arrays as argument so that ncomps can be deduced
     *
     * \param[in] lev level of fields
     * \param[in] islice_dst destination slice
     * \param[in] comps_dst array of destination component names
     * \param[in] islice_src source slice
     * \param[in] comps_src array of source component names
     */
    template<int ncomps>
    void duplicate (const int lev,
                    const int islice_dst, const char * const (&comps_dst)[ncomps],
                    const int islice_src, const char * const (&comps_src)[ncomps]) {
        amrex::GpuArray<int, ncomps> c_idx_src = {};
        amrex::GpuArray<int, ncomps> c_idx_dst = {};
        for (int i=0; i<ncomps; ++i) {
            c_idx_src[i] = Comps[islice_src][comps_src[i]];
            c_idx_dst[i] = Comps[islice_dst][comps_dst[i]];
        }
        amrex::MultiFab& mfab = getSlices(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mfab, DfltMfiTlng); mfi.isValid(); ++mfi) {
            const Array3<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int)
                {
                    for (int n=0; n<ncomps; ++n) {
                        array(i,j,c_idx_dst[n]) = array(i,j,c_idx_src[n]);
                    }
                });
        }
    }

    /** \brief add all selected fields between slices or on the same slice
     * Uses references to C-arrays as argument so that ncomps can be deduced
     *
     * \param[in] lev level of fields
     * \param[in] islice_dst destination slice
     * \param[in] comps_dst array of destination component names
     * \param[in] islice_src source slice
     * \param[in] comps_src array of source component names
     */
    template<int ncomps>
    void add (const int lev,
              const int islice_dst, const char * const (&comps_dst)[ncomps],
              const int islice_src, const char * const (&comps_src)[ncomps]) {
        amrex::GpuArray<int, ncomps> c_idx_src = {};
        amrex::GpuArray<int, ncomps> c_idx_dst = {};
        for (int i=0; i<ncomps; ++i) {
            c_idx_src[i] = Comps[islice_src][comps_src[i]];
            c_idx_dst[i] = Comps[islice_dst][comps_dst[i]];
        }
        amrex::MultiFab& mfab = getSlices(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mfab, DfltMfiTlng); mfi.isValid(); ++mfi) {
            const Array3<amrex::Real> array = mfab.array(mfi);
            // only one Kernel for all fields
            amrex::ParallelFor(mfi.growntilebox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int)
                {
                    for (int n=0; n<ncomps; ++n) {
                        array(i,j,c_idx_dst[n]) += array(i,j,c_idx_src[n]);
                    }
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
        amrex::MultiFab& mfab = getSlices(lev);

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
    inline static amrex::IntVect m_slices_nguards {-1, -1, -1};
    /** Number of guard cells for poisson solver MultiFab */
    inline static amrex::IntVect m_poisson_nguards {-1, -1, -1};
    /** If the poisson solver should include the guard cells */
    bool m_extended_solve = false;
    /** Whether the currents should be symmetrized for the field solve */
    bool m_do_symmetrize = false;

private:
    /** Vector over levels of all required fields to compute current slice */
    amrex::Vector<amrex::MultiFab> m_slices;
    /** Type of poisson solver to use */
    std::string m_poisson_solver_str = "FFTDirichlet";
    /** Class to handle transverse FFT Poisson solver on 1 slice */
    amrex::Vector<std::unique_ptr<FFTPoissonSolver>> m_poisson_solver;
    /** Temporary density arrays. one per OpenMP thread, used when tiling is on. */
    amrex::Vector<amrex::FArrayBox> m_tmp_densities;
    /** Stores temporary values for z interpolation in Fields::Copy */
    amrex::Gpu::DeviceVector<amrex::Real> m_rel_z_vec;
    /** Stores temporary values for z interpolation in Fields::Copy on the CPU */
    amrex::Gpu::PinnedVector<amrex::Real> m_rel_z_vec_cpu;
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
    /** Number of real field properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nrp = 9;
    /** How often the insitu field diagnostics should be computed and written
     * Default is 0, meaning no output */
    int m_insitu_period {0};
    /** All per-slice real field properties */
    amrex::Vector<amrex::Real> m_insitu_rdata;
    /** Sum of all per-slice real field properties */
    amrex::Vector<amrex::Real> m_insitu_sum_rdata;
    /** Prefix/path for the output files */
    std::string m_insitu_file_prefix = "diags/field_insitu";
};

#endif
