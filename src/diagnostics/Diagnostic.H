/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef DIAGNOSTIC_H_
#define DIAGNOSTIC_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <vector>

/** type of diagnostics: full xyz array or xz slice or yz slice */
enum struct DiagType{xyz, xz, yz};

/** \brief This class holds data for 1 diagnostics (full or slice) */
class Diagnostic
{

public:

    /** \brief Constructor */
    explicit Diagnostic (int nlev);

    /** \brief Determine which data to output */
    void Initialize (const int lev, bool do_laser);

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

    /** \brief return the number of Field components to output */
    int getTotalNFields() { return m_nfields + 2*m_do_laser; }

    /** \brief return names of the beams to output */
    amrex::Vector<std::string>& getBeamNames () { return m_output_beam_names; }

    /** return the diagnostics geometry */
    amrex::Vector<amrex::Geometry>& getGeom () { return m_geom_io; }

    /** return slice direction of the diagnostics */
    int sliceDir () {return m_slice_dir;}

    /** Whether to output laser */
    bool doLaser () const {return m_do_laser;}

    /**< return if there is field output to write */
    std::vector<bool>& hasField () {return m_has_field;}

    /** return coarsening ratio of diagnostics */
    amrex::IntVect getCoarsening (int lev) { return m_diag_coarsen[lev]; }

    /** \brief calculate box which possibly was trimmed in case of slice IO
     *
     * \param[in,out] box_3d local box to be possibly trimmed to a slice box
     * \param[in,out] domain_3d domain box to be possibly trimmed to a slice box
     * \param[in,out] rbox_3d real box to be possibly trimmed to a slice box
     */
    void TrimIOBox (amrex::Box& box_3d, amrex::Box& domain_3d, amrex::RealBox& rbox_3d);

    /** \brief resizes the FArrayBox of the diagnostics to the currently calculated box
     *
     * \param[in] local_box box to which the FArrayBox of the diagnostics will be resized to
     * \param[in] domain box to which the Geometry of the diagnostics will be resized to
     * \param[in] lev MR level
     * \param[in] geom geometry of the full simulation domain
     */
    void ResizeFDiagFAB (amrex::Box local_box, amrex::Box domain, const int lev,
                         amrex::Geometry const& geom);

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
    bool m_do_laser {false}; /**< Whether to output the laser */
    amrex::Vector<amrex::Geometry> m_geom_io; /**< Diagnostics geometry */
    bool m_include_ghost_cells = false; /**< if ghost cells are included in output */
    bool m_initialized = false; /**< if this object is fully initialized */
    std::vector<bool> m_has_field; /**< if there is field output to write */
};

#endif // DIAGNOSTIC_H_
