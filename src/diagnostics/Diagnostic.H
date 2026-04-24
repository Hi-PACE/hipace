/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef DIAGNOSTIC_H_
#define DIAGNOSTIC_H_

#include "utils/IOUtil.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <vector>

class Fields;
class MultiBeam;
class MultiLaser;
class MultiPlasma;


/** \brief This struct holds data for one field diagnostic on one MR level */
struct DiagnosticData
{
    std::string m_diag_name; /**< name used for input parameters and in the output */
    enum struct diag_type {
        field,
        laser,
        histogram
    } m_base_diag_type = diag_type::field;  /**< which geometry the diagnostics is based on */
    /** Number of iterations between consecutive output dumps.
     * Default is 0, meaning no output */
    utils::DiagPeriod m_output_period = "0";
    bool m_has_output = false; /**< if there is output to write */
    int m_level = 0; /**< MR level */
    amrex::IntVect m_remove_axis {0, 0, 0}; /**< which axis should be removed */
    amrex::Vector<std::string> m_axis_labels {}; /**< axis labels in C order */
    bool m_integrate_along_z = false; /**< resolve or integrate along zeta */
    amrex::Geometry m_geom_io; /**< Diagnostics geometry of the openPMD output */
    amrex::Geometry m_realspace_geom; /**< Diagnostics geometry in simulation realspace */
    bool m_include_ghost_cells = false; /**< if ghost cells are included in output */
    bool m_use_custom_size_lo = false; /**< if a user defined diagnostics size should be used (lo)*/
    bool m_use_custom_size_hi = false; /**< if a user defined diagnostics size should be used (hi)*/
    /** 3D array with lower ends of the diagnostics grid */
    amrex::RealVect m_diag_lo {0., 0., 0.};
    /** 3D array with upper ends of the diagnostics grid */
    amrex::RealVect m_diag_hi {0., 0., 0.};
    amrex::IntVect m_diag_coarsen {1, 1, 1}; /**< xyz coarsening ratio (positive) */
    int m_nfields = 0; /**< Number of physical fields to write */
    amrex::Vector<std::string> m_comps_output; /**< Component names to Write to output file */
    /** Component indexes to Write to output file */
    amrex::Gpu::Buffer<int> m_comps_output_idx;
    /** All fields */
    amrex::FArrayBox m_F_real;
    /** FAB for laser */
    amrex::BaseFab<amrex::GpuComplex<amrex::Real>> m_F_complex;

    // histogram

    amrex::Vector<std::string> m_hist_species_names; /**< beam and plasma names for histogram */
    int m_hist_num_dims = 1; /**< 1D or 2D histogram */
    amrex::Vector<int> m_hist_num_bins; /**< number of bins per hist axis */
    amrex::Vector<amrex::Real> m_hist_bins_lo; /**< bounds for each hist axis */
    amrex::Vector<amrex::Real> m_hist_bins_hi; /**< bounds for each hist axis */
    amrex::FArrayBox m_hist_gpu_fab; /**< hist slice data cache on GPU */
    /** parser funcitons for histogram */
    amrex::Parser m_hist_parser_q1;
    amrex::ParserExecutor<9> m_hist_exe_q1;
    amrex::Parser m_hist_parser_q2;
    amrex::ParserExecutor<9> m_hist_exe_q2;
    amrex::Parser m_hist_parser_w;
    amrex::ParserExecutor<9> m_hist_exe_w;
};


/** \brief This class holds data for all diagnostics */
class Diagnostic
{

public:

    /** Read parameters from the input file */
    void ReadParameters (int nlev, bool use_laser);

    /** \brief Determine which data to output */
    void Initialize (int nlev, bool use_laser);

    /** \brief return names of the beams to output */
    amrex::Vector<std::string>& getBeamNames () { return m_output_beam_names; }

    /** \brief return data for all field diagnostics */
    amrex::Vector<DiagnosticData>& getDiagData () { return m_diag_data; }

    /** \brief determines if a single field diagnostic has any output on in this time step
     *
     * \param[in] fd field diagnostic
     * \param[in] output_step current step index
     * \param[in] output_time physical time of the current step
     * \param[in] is_last_step if this is the last time step of the simulation
     */
    static bool hasFieldOutput (const DiagnosticData& fd, int output_step,
                                amrex::Real output_time, bool is_last_step)
    {
        return (fd.m_nfields > 0) &&
            fd.m_output_period.doDiagnostics(output_step, output_time, is_last_step);
    }

    /** \brief determines if any field diagnostic has any output on in this time step
     *
     * \param[in] output_step current step index
     * \param[in] output_time physical time of the current step
     * \param[in] is_last_step if this is the last time step of the simulation
     */
    [[nodiscard]] bool
    hasAnyFieldOutput (int output_step, amrex::Real output_time, bool is_last_step) const
    {
        for (const auto& fd : m_diag_data) {
            if (hasFieldOutput(fd, output_step, output_time, is_last_step)) return true;
        }
        return false;
    }

    /** \brief determines if any beam diagnostic has any output on in this time step
     *
     * \param[in] output_step current step index
     * \param[in] output_time physical time of the current step
     * \param[in] is_last_step if this is the last time step of the simulation
     */
    [[nodiscard]] bool
    hasBeamOutput (int output_step, amrex::Real output_time, bool is_last_step) const
    {
        return m_output_beam_names.size() > 0 &&
            m_beam_output_period.doDiagnostics(output_step, output_time, is_last_step);
    }

    /** \brief determines if any field or beam diagnostic has any output on in this time step
     *
     * \param[in] output_step current step index
     * \param[in] output_time physical time of the current step
     * \param[in] is_last_step if this is the last time step of the simulation
     */
    [[nodiscard]] bool
    hasAnyOutput (int output_step, amrex::Real output_time, bool is_last_step) const
    {
        return hasAnyFieldOutput(output_step, output_time, is_last_step) ||
               hasBeamOutput(output_step, output_time, is_last_step);
    }

    /** \brief determines if rho is requested as a diagnostic
     */
    [[nodiscard]] bool
    needsRho () const;

    /** \brief determines if rho for every individual plasma is requested as a diagnostic
     */
    [[nodiscard]] bool
    needsRhoIndividual () const;

    /** \brief determines if the temperature parameters for every individual plasma are requested as a diagnostic
     */
    [[nodiscard]] bool
    needsTempIndividual () const;

    /** \brief resizes the FArrayBox of the diagnostics to the currently calculated box
     *
     * \param[in] field_geom geometry of the full simulation domain for all field MR levels
     * \param[in] laser_geom geometry of the full simulation domain for the laser
     * \param[in] output_step current step index
     * \param[in] output_time physical time of the current step
     * \param[in] is_last_step if this is the last time step of the simulation
     */
    void ResizeFDiagFAB (amrex::Vector<amrex::Geometry>& field_geom,
                         amrex::Geometry const& laser_geom, int output_step,
                         amrex::Real output_time, bool is_last_step);

    /** \brief calculate where data from slice islice will go to using zeroth-order interpolation
     * returns if this slice is needed and the destination slice index
     *
     * \param[in] fd diag data object
     * \param[in] islice current slice
     * \param[in] geom3d 3d domain used for the z direction
     */
    static std::pair<bool, int>
    ReverseShapeFactor (const DiagnosticData& fd, int islice, const amrex::Geometry& geom3d);

    /** \brief Fill diagnostic arrays with the simulation data available on the current slice
     *
     * \param[in] islice current slice
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] fields Fields object
     * \param[in] lasers MultiLaser object
     * \param[in] plasmas MultiPlasma object
     * \param[in] beams MultiBeam object
     * \param[in] field_geom geometry for all field MR levels
     */
    void
    FillDiagnostics (int islice, int current_N_level,
                     Fields& fields, MultiLaser& lasers,
                     MultiPlasma& plasmas, MultiBeam& beams,
                     const amrex::Vector<amrex::Geometry>& field_geom);

private:
    amrex::Vector<std::string> m_output_beam_names; /**< Component names to Write to output file */
    /** Number of iterations between consecutive output dumps.
     * Default is 0, meaning no output */
    utils::DiagPeriod m_beam_output_period = "0";
    amrex::Vector<DiagnosticData> m_diag_data; /**< individual field diag data */
};

#endif // DIAGNOSTIC_H_
