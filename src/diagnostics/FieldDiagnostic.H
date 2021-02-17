#ifndef FIELDDIAGNOSTIC_H_
#define FIELDDIAGNOSTIC_H_

#include <AMReX_MultiFab.H>

/** type of diagnostics: full xyz array or xz slice or yz slice */
enum struct DiagType{xyz, xz, yz};

/** \brief This class holds data for 1 diagnostics (full or slice) */
class FieldDiagnostic
{

public:

    /** \brief Constructor */
    explicit FieldDiagnostic (int nlev);

    /** \brief allocate arrays of this MF
     *
     * \param[in] lev MR level
     * \param[in] ba BoxArray of the full simulation domain
     * \param[in] nfields number of field components
     * \param[in] dm DistributionMapping of the full simulation domain
     * \param[in] geom geometry of the full simulation domain
     */
    void AllocData (int lev, const amrex::BoxArray& ba, int nfields,
                    const amrex::DistributionMapping& dm, amrex::Geometry const& geom);

    /** \brief return the main diagnostics multifab */
    amrex::Vector<amrex::MultiFab>& getF () { return m_F; };

    /** \brief return the main diagnostics multifab
     *
     * \param[in] lev MR level
     */
    amrex::MultiFab& getF (int lev) {return m_F[lev];};

    /** \brief return Component names of Fields to output */
    amrex::Vector<std::string>& getComps () { return m_comps_output; };

    /** return the diagnostics geometry */
    amrex::Vector<amrex::Geometry>& getGeom () { return m_geom_io; };

    /** return slice direction of the diagnostics */
    int sliceDir () {return m_slice_dir;};

private:

    /** Vector over levels, all fields */
    amrex::Vector<amrex::MultiFab> m_F;
    DiagType m_diag_type; /**< Type of diagnostics (xyz xz yz) */
    int m_slice_dir; /**< Slicing direction */
    amrex::Vector<std::string> m_comps_output; /**< Component names to Write to output file */
    int m_nfields; /**< Number of physical fields to write */
    amrex::Vector<amrex::Geometry> m_geom_io; /**< Diagnostics geometry */
};

#endif // FIELDDIAGNOSTIC_H_
