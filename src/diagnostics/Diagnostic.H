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



struct FieldDiagnosticData
{
    std::string m_diag_name;

    int m_level = 0;

    int m_slice_dir; /**< Slicing direction */

    bool m_include_ghost_cells = false; /**< if ghost cells are included in output */

    bool m_use_custom_size_lo = false; /**< if a user defined diagnostics size should be used (lo)*/
    bool m_use_custom_size_hi = false; /**< if a user defined diagnostics size should be used (hi)*/
    /** 3D array with lower ends of the diagnostics grid */
    amrex::Array<amrex::Real, 3> m_diag_lo {0., 0., 0.};
    /** 3D array with upper ends of the diagnostics grid */
    amrex::Array<amrex::Real, 3> m_diag_hi {0., 0., 0.};

    amrex::IntVect m_diag_coarsen; /**< xyz coarsening ratio (positive) */




    bool m_do_laser {false}; /**< Whether to output the laser */

    int m_nfields; /**< Number of physical fields to write */
    amrex::Vector<std::string> m_comps_output; /**< Component names to Write to output file */
    /** Component indexes to Write to output file */
    amrex::Gpu::DeviceVector<int> m_comps_output_idx;


    /** Vector over levels, all fields */
    amrex::FArrayBox m_F;

    amrex::Geometry m_geom_io; /**< Diagnostics geometry */

    bool m_has_field; /**< if there is field output to write */

    /** Number of iterations between consecutive output dumps.
     * Default is -1, meaning no output */
    int m_output_period = -1;
};


/** \brief This class holds data for 1 diagnostics (full or slice) */
class Diagnostic
{

public:

    /** \brief Constructor */
    explicit Diagnostic (int nlev);

    /** \brief Determine which data to output */
    void Initialize (const int lev, bool do_laser);

    /** \brief return names of the beams to output */
    amrex::Vector<std::string>& getBeamNames () { return m_output_beam_names; }


    amrex::Vector<FieldDiagnosticData>& getFieldData () { return m_field_data; }


    static bool hasFieldOutput (const FieldDiagnosticData& fd,
                         int output_step, int max_step,
                         amrex::Real output_time, amrex::Real max_time)
    {
        return fd.m_output_period >= 0 && (
            (output_time == max_time) ||
            (output_step == max_step) ||
            (output_step % fd.m_output_period == 0) );
    }

    bool hasAnyFieldOutput (int output_step, int max_step,
                            amrex::Real output_time, amrex::Real max_time) const
    {
        for (const auto& fd : m_field_data) {
            if (hasFieldOutput(fd, output_step, max_step, output_time, max_time)) return true;
        }
        return false;
    }

    bool hasBeamOutput (int output_step, int max_step,
                        amrex::Real output_time, amrex::Real max_time) const
    {
        return m_beam_output_period >= 0 && (
            (output_time == max_time) ||
            (output_step == max_step) ||
            (output_step % m_beam_output_period == 0) );
    }

    bool hasAnyOutput (int output_step, int max_step,
                       amrex::Real output_time, amrex::Real max_time) const
    {
        return hasAnyFieldOutput(output_step, max_step, output_time, max_time) ||
               hasBeamOutput(output_step, max_step, output_time, max_time);
    }


    /** \brief calculate box which possibly was trimmed in case of slice IO
     *
     * \param[in] slice_dir slicing direction
     * \param[in,out] box_3d local box to be possibly trimmed to a slice box
     * \param[in,out] domain_3d domain box to be possibly trimmed to a slice box
     * \param[in,out] rbox_3d real box to be possibly trimmed to a slice box
     */
    void TrimIOBox (int slice_dir, amrex::Box& box_3d,
                    amrex::Box& domain_3d, amrex::RealBox& rbox_3d);

    /** \brief resizes the FArrayBox of the diagnostics to the currently calculated box
     *
     * \param[in] a_local_box box to which the FArrayBox of the diagnostics will be resized to
     * \param[in] a_domain box to which the Geometry of the diagnostics will be resized to
     * \param[in] lev MR level
     * \param[in] geom geometry of the full simulation domain
     */
    void ResizeFDiagFAB (const amrex::Box a_local_box, const amrex::Box a_domain, const int lev,
                         amrex::Geometry const& geom, int output_step, int max_step,
                         amrex::Real output_time, amrex::Real max_time);

private:

    amrex::Vector<std::string> m_output_beam_names; /**< Component names to Write to output file */
    int m_beam_output_period = -1;

    bool m_initialized = false; /**< if this object is fully initialized */

    amrex::Vector<FieldDiagnosticData> m_field_data;

};

#endif // DIAGNOSTIC_H_
