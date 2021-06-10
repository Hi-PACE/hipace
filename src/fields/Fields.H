#ifndef FIELDS_H_
#define FIELDS_H_

#include "fft_poisson_solver/FFTPoissonSolver.H"
#include "diagnostics/Diagnostic.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

class Hipace;

/** \brief Direction of field copies: from 3D F to 2D slice S or the other way round */
enum struct FieldCopyType { FtoS, StoF };

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
                {"Bx", 0}, {"By", 1}, {"jx", 2}, {"jx_beam", 3}, {"jy", 4}, {"jy_beam", 5}, {"N", 6}
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

/** \brief Operation performed in the TransverseDerivative function:
 * either assign or add to the destination array
 */
enum struct SliceOperatorType { Assign, Add };

/** \brief Direction of each dimension. Can be used for clean handling 2D vs. 3D in the future */
struct Direction{
    enum dir{x=0, y, z};
};

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
     * \param[in] ba BoxArray for the simulation
     * \param[in] dm DistributionMapping for the whole simulation
     * \param[in] geom Geometry
     * \param[in] slice_ba BoxArray for the slice
     * \param[in] slice_dm DistributionMapping for the slice
     */
    void AllocData (
        int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
        amrex::Geometry const& geom, const amrex::BoxArray& slice_ba,
        const amrex::DistributionMapping& slice_dm);

    /** Class to handle transverse FFT Poisson solver on 1 slice */
    std::unique_ptr<FFTPoissonSolver> m_poisson_solver;
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

    /** \brief Copy between the full FArrayBox and slice MultiFab.
     *
     * \param[in] lev MR level
     * \param[in] i_slice z slice in which to write the data
     * \param[in] copy_type whether to copy from the xy slice to the main array or opposite
     * \param[in] slice_comp first component of the xy slice to copy from/to
     * \param[in] full_comp first component of the full array to copy from/to
     * \param[in] ncomp number of components to copy
     * \param[in,out] fab full FArrayBox
     * \param[in] slice_dir slicing direction. 0=x, 1=y, -1=no slicing (full 3D)
     * \param[in] geom main geometry
     */
    void Copy (int lev, int i_slice, FieldCopyType copy_type, int slice_comp, int full_comp,
               int ncomp, amrex::FArrayBox& fab, int slice_dir, amrex::Geometry geom);

    /** \brief Shift slices by 1 element: slices (1,2) are then stored in (2,3).
     *
     * When looping over slices from head to tail, the same slice MultiFabs are used
     * to compute each slice. The current slice is always stored in index 1.
     * Hence, after one slice is computed, slices must be shifted by 1 element.
     *
     * \param[in] lev MR level
     */
    void ShiftSlices (int lev);

    /** add rho of the ions to rho (this slice) */
    void AddRhoIons (const int lev, bool inverse=false);

    /** \brief Adds the beam currents jx_beam, jy_beam, jz_beam to the general currents jx, jy, jz
     *
     * \param[in] lev current level
     * \param[in] which_slice current slice, either WhichSlice::This or WhichSlice::Next.
     */
    void AddBeamCurrents (const int lev, const int which_slice);

    /** Compute transverse derivative of 1 slice*/
    void TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst,
                               const int direction, const amrex::Real dx,
                               const amrex::Real mult_coeff=1.,
                               const SliceOperatorType slice_operator=SliceOperatorType::Assign,
                               const int scomp=0, const int dcomp=0);

    /** Compute longitudinal derivative (difference between two slices) */
    void LongitudinalDerivative (const amrex::MultiFab& src, const amrex::MultiFab& src2,
                                 amrex::MultiFab& dst, const amrex::Real dz,
                                 const amrex::Real mult_coeff=1.,
                                 const SliceOperatorType slice_operator=SliceOperatorType::Assign,
                                 const int sc1omp=0, const int s2comp=0, const int dcomp=0);

    /** \brief Compute ExmBy and EypBx on the slice container from J by solving a Poisson equation
     * ExmBy and EypBx are solved in the same function because both rely on Psi.
     *
     * \param[in] geom Geometry
     * \param[in] m_comm_xy transverse communicator on the slice
     * \param[in] lev current level
     */
    void SolvePoissonExmByAndEypBx (amrex::Geometry const& geom, const MPI_Comm& m_comm_xy,
                                    const int lev);
    /** \brief Compute Ez on the slice container from J by solving a Poisson equation
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     */
    void SolvePoissonEz (amrex::Geometry const& geom, const int lev);
    /** \brief Compute Bx on the slice container from J by solving a Poisson equation
     *
     * \param[in,out] Bx_iter Bx field during current iteration of the predictor-corrector loop
     * \param[in] geom Geometry
     * \param[in] lev current level
     */
    void SolvePoissonBx (amrex::MultiFab& Bx_iter, amrex::Geometry const& geom, const int lev);
    /** \brief Compute By on the slice container from J by solving a Poisson equation
     *
     * \param[in,out] By_iter By field during current iteration of the predictor-corrector loop
     * \param[in] geom Geometry
     * \param[in] lev current level
     */
    void SolvePoissonBy (amrex::MultiFab& By_iter,amrex::Geometry const& geom, const int lev);
    /** \brief Compute Bz on the slice container from J by solving a Poisson equation
     *
     * \param[in] geom Geometry
     * \param[in] lev current level
     */
    void SolvePoissonBz (amrex::Geometry const& geom, const int lev);
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

private:
    /** Vector over levels, array of 4 slices required to compute current slice */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>> m_slices;
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards {-1, -1, -1};
    /** Whether to use Dirichlet BC for the Poisson solver. Otherwise, periodic */
    bool m_do_dirichlet_poisson = true;
};

#endif
