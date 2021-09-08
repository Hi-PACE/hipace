#ifndef DIAGNOSTIC_H_
#define DIAGNOSTIC_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

/** type of diagnostics: full xyz array or xz slice or yz slice */
enum struct DiagType{xyz, xz, yz};

/** \brief This class holds data for 1 diagnostics (full or slice) */
class Diagnostic
{

public:

    /** \brief Constructor */
    explicit Diagnostic (int nlev);

    /** \brief allocate arrays of this MF
     *
     * \param[in] lev MR level
     * \param[in] bx Box for initialization
     * \param[in] geom geometry of the full simulation domain
     */
    void AllocData (int lev, const amrex::Box& bx, amrex::Geometry const& geom);

    /** \brief return the main diagnostics multifab */
    amrex::Vector<amrex::FArrayBox>& getF () { return m_F; }

    /** \brief return the main diagnostics multifab
     *
     * \param[in] lev MR level
     */
    amrex::FArrayBox& getF (int lev) { return m_F[lev]; }

    /** \brief return Component names of Fields to output */
    amrex::Vector<std::string>& getComps () { return m_comps_output; }

    /** \brief return Component indexes of Fields to output */
    amrex::Gpu::DeviceVector<int>& getCompsIdx() { return m_comps_output_idx; }

    /** \brief return the number of Field components to output */
    int getNFields() { return m_nfields; }

    /** \brief return names of the beams to output */
    amrex::Vector<std::string>& getBeamNames () { return m_output_beam_names; }

    /** return the diagnostics geometry */
    amrex::Vector<amrex::Geometry>& getGeom () { return m_geom_io; }

    /** return slice direction of the diagnostics */
    int sliceDir () {return m_slice_dir;}

    /** return coarsening ratio of diagnostics */
    amrex::IntVect getCoarsening (int lev) { return m_diag_coarsen[lev]; }

    /** \brief return box which possibly was trimmed in case of slice IO
     *
     * \param[in] box_3d box to be possibly trimmed to a slice box
     */
    amrex::Box TrimIOBox (const amrex::Box box_3d);

    /** \brief resizes the FArrayBox of the diagnostics to the currently calculated box
     *
     * \param[in] box box to which the FArrayBox of the diagnostics will be resized to
     * \param[in] lev MR level
     */
    void ResizeFDiagFAB (const amrex::Box box, const int lev);

private:

    /** Vector over levels, all fields */
    amrex::Vector<amrex::FArrayBox> m_F;
    DiagType m_diag_type; /**< Type of diagnostics (xyz xz yz) */
    int m_slice_dir; /**< Slicing direction */
    amrex::Vector<amrex::IntVect> m_diag_coarsen; /**< xyz coarsening ratio (positive) */
    amrex::Vector<std::string> m_comps_output; /**< Component names to Write to output file */
    /** Component indexes to Write to output file */
    amrex::Gpu::DeviceVector<int> m_comps_output_idx;
    amrex::Vector<std::string> m_output_beam_names; /**< Component names to Write to output file */
    int m_nfields; /**< Number of physical fields to write */
    amrex::Vector<amrex::Geometry> m_geom_io; /**< Diagnostics geometry */
};

#endif // DIAGNOSTIC_H_
