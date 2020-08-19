#ifndef FIELDS_H_
#define FIELDS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

class Hipace;

/** \brief Map names and indices of each fields in the field Multifabs (both 3D and 2D)
 */
struct FieldComps{
    enum Comps{ExmBy = 0, EypBx, Ez, Bx, By, Bz, jx, jy, jz, nfields};
};

/** \brief Direction of field copies: from 3D F to 2D slice S or the other way round */
enum struct FieldCopyType { FtoS, StoF };

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
    static constexpr int m_nslices = 4;

public:
    explicit Fields (Hipace const* a_hipace);

    /** Allocate MultiFabs for the 3D array and the 2D slices
     * and define the BoxArrays and DistributionMappings.
     * \param[in] lev MR level
     * \param[in] ba BoxArray for the simulation
     * \param[in] dm DistributionMapping for the whole simulation
     */
    void AllocData (int lev, const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm);

    /** get function for the main 3D array F */
    amrex::Vector<amrex::MultiFab>& getF () { return m_F; }
    /** get function for the main 3D array F
     * \param lev MR level */
    amrex::MultiFab& getF (int lev) { return m_F[lev]; }
    /** get function for the main 3D array F
     * \param lev MR level
     * \param lev icomp component of the field */
    amrex::MultiFab getF (int lev, int icomp );

    /** get function for the 2D slices */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>>& getSlices () {return m_slices; }
    /** get function for the 2D slices
     * \param[in] lev MR level
     */
    std::array<amrex::MultiFab, m_nslices>& getSlices (int lev) {return m_slices[lev]; }
    /** get function for the 2D slices
     * \param[in] lev MR level
     * \param[in] i slice index (currently from 0 to 3 )
     */
    amrex::MultiFab& getSlices (int lev, int i) {return m_slices[lev][i]; }

    /** Copy between the full MultiFab and slice MultiFab */
    void Copy (int lev, int i_slice, FieldCopyType copy_type, int slice_comp, int full_comp,
               int ncomp);

    /** \brief Shift slices by 1 element: slices (1,2) are then stored in (2,3).
     *
     * When looping over slices from head to tail, the same slice MultiFabs are used
     * to compute each slice. The current slice is always stored in index 1.
     * Hence, after one slice is computed, slices must be shifted by 1 element.
     *
     * \param[in] lev MR level
     */
    void ShiftSlices (int lev);

    /** Compute transverse derivative of 1 slice*/
    void TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                               const amrex::Real dx, const int scomp=0, const int dcomp=0);

private:
    /** Pointer to const singleton of class Hipace */
    Hipace const* m_hipace;
    /** Vector over levels, all fields */
    amrex::Vector<amrex::MultiFab> m_F;
    /** Vector over levels, array of 4 slices required to compute current slice */
    amrex::Vector<std::array<amrex::MultiFab, m_nslices>> m_slices;
    /** Number of guard cells for main MultiFab */
    amrex::IntVect m_nguards {-1, -1, -1};
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards {-1, -1, -1};
};

#endif
