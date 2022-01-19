#ifndef FIELDS_H_
#define FIELDS_H_

#include "fft_poisson_solver/FFTPoissonSolver.H"
#include "diagnostics/Diagnostic.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

class Hipace;

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichSlice {
    enum slice { Next=0, This, Previous1, Previous2, RhoIons, N };
};

/** \brief Map names and indices of each fields in each slice
 */
static std::array<std::map<std::string, int>, 5> Comps
{{
        /* WhichSlice::Next */
        {{
                {"jx", 0}, {"jx_beam", 1}, {"jy", 2}, {"jy_beam", 3}, {"N", 4}
            }},
        /* WhichSlice::This */
        {{
                {"ExmBy", 0}, {"EypBx", 1}, {"Ez", 2}, {"Bx", 3}, {"By", 4}, {"Bz", 5}, {"jx", 6},
                {"jx_beam", 7}, {"jy", 8}, {"jy_beam", 9}, {"jz", 10}, {"jz_beam", 11}, {"rho", 12},
                {"Psi", 13}, {"jxx", 14}, {"jxy", 15}, {"jyy", 16}, {"N", 17}
            }},
        /* WhichSlice::Previous1 */
        {{
                {"Ez", 0}, {"Bx", 1}, {"By", 2}, {"Bz", 3}, {"jx", 4}, {"jx_beam", 5}, {"jy", 6},
                {"jy_beam", 7}, {"rho", 8}, {"Psi", 9}, {"N", 10}
            }},
        /* WhichSlice::Previous2 */
        {{
                {"Bx", 0}, {"By", 1}, {"N", 2}
            }},
        /* WhichSlice::RhoIons */
        {{
                {"rho", 0}, {"N", 1}
            }}
    }};

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
     * \param[in] slice_comp first component of the xy slice to copy from/to
     * \param[in] full_comp first component of the full array to copy from/to
     * \param[in] diag_comps_vect the field components to copy
     * \param[in] ncomp number of components to copy
     * \param[in,out] fab full FArrayBox
     * \param[in] slice_dir slicing direction. 0=x, 1=y, -1=no slicing (full 3D)
     * \param[in] diag_coarsen coarsening ratio of diagnostics
     * \param[in] geom main geometry
     */
    void Copy (const int lev, const int i_slice, const int slice_comp, const int full_comp,
               const amrex::Gpu::DeviceVector<int>& diag_comps_vect,
               const int ncomp, amrex::FArrayBox& fab, const int slice_dir,
               const amrex::IntVect diag_coarsen, const amrex::Geometry geom);

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

    /** \brief Adds the beam currents jx_beam, jy_beam, jz_beam to the general currents jx, jy, jz
     *
     * \param[in] lev current level
     * \param[in] which_slice current slice, either WhichSlice::This or WhichSlice::Next.
     */
    void AddBeamCurrents (const int lev, const int which_slice);

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

    /** Number of guard cells for slices MultiFab */
    static amrex::IntVect m_slices_nguards;
    /** Number of guard cells for poisson solver MultiFab */
    static amrex::IntVect m_poisson_nguards;
private:
    /** Vector over levels, array of 4 slices required to compute current slice */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>> m_slices;
    /** Whether to use Dirichlet BC for the Poisson solver. Otherwise, periodic */
    bool m_do_dirichlet_poisson = true;
    /** Temporary density arrays. one per OpenMP thread, used when tiling is on. */
    amrex::Vector<amrex::FArrayBox> m_tmp_densities;
};

#endif
