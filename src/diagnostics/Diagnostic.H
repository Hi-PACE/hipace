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

// 16-byte alignment enables coalesced memory access which is significantly faster
struct alignas(2*sizeof(amrex::Real)) AlignedComplex {
    amrex::Real real;
    amrex::Real imag;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    AlignedComplex& operator+= (const AlignedComplex& rhs) {
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }
};


/** \brief This struct holds data for one field diagnostic on one MR level */
struct FieldDiagnosticData
{
    std::string m_diag_name; /**< name used for input parameters and in the output */
    int m_level = 0; /**< MR level */
    int m_slice_dir; /**< Slicing direction */
    bool m_include_ghost_cells = false; /**< if ghost cells are included in output */
    bool m_use_custom_size_lo = false; /**< if a user defined diagnostics size should be used (lo)*/
    bool m_use_custom_size_hi = false; /**< if a user defined diagnostics size should be used (hi)*/
    /** 3D array with lower ends of the diagnostics grid */
    amrex::RealVect m_diag_lo {0., 0., 0.};
    /** 3D array with upper ends of the diagnostics grid */
    amrex::RealVect m_diag_hi {0., 0., 0.};
    amrex::IntVect m_diag_coarsen; /**< xyz coarsening ratio (positive) */
    bool m_do_laser {false}; /**< Whether to output the laser */
    int m_nfields; /**< Number of physical fields to write */
    amrex::Vector<std::string> m_comps_output; /**< Component names to Write to output file */
    /** Component indexes to Write to output file */
    amrex::Gpu::DeviceVector<int> m_comps_output_idx;
    /** Vector over levels, all fields */
    amrex::FArrayBox m_F;
    using complex_type = AlignedComplex;
    /** FAB for laser */
    amrex::BaseFab<complex_type> m_F_laser;
    amrex::Geometry m_geom_io; /**< Diagnostics geometry */
    bool m_has_field; /**< if there is field output to write */
    /** Number of iterations between consecutive output dumps.
     * Default is 0, meaning no output */
    int m_output_period = 0;
    /** Name of the laser in input and output files */
    inline static std::string m_laser_io_name = "laserEnvelope";
};


/** \brief This class holds data for all diagnostics */
class Diagnostic
{

public:

    /** \brief Constructor */
    explicit Diagnostic (int nlev);

    /** \brief Determine which data to output */
    void Initialize (const int lev, bool do_laser);

    /** \brief return names of the beams to output */
    amrex::Vector<std::string>& getBeamNames () { return m_output_beam_names; }

    /** \brief return data for all field diagnostics */
    amrex::Vector<FieldDiagnosticData>& getFieldData () { return m_field_data; }

    /** \brief determines if a single field diagnostic has any output on in this time step
     *
     * \param[in] fd field diagnostic
     * \param[in] output_step current step index
     * \param[in] max_step last step index
     * \param[in] output_time physical time of current step
     * \param[in] max_time physical time of last step
     */
    static bool hasFieldOutput (const FieldDiagnosticData& fd,
                         int output_step, int max_step,
                         amrex::Real output_time, amrex::Real max_time)
    {
        return  (fd.m_nfields > 0 || fd.m_do_laser) &&
            utils::doDiagnostics(fd.m_output_period, output_step, max_step,
                            output_time, max_time);
    }

    /** \brief determines if any field diagnostic has any output on in this time step
     *
     * \param[in] output_step current step index
     * \param[in] max_step last step index
     * \param[in] output_time physical time of current step
     * \param[in] max_time physical time of last step
     */
    bool hasAnyFieldOutput (int output_step, int max_step,
                            amrex::Real output_time, amrex::Real max_time) const
    {
        for (const auto& fd : m_field_data) {
            if (hasFieldOutput(fd, output_step, max_step, output_time, max_time)) return true;
        }
        return false;
    }

    /** \brief determines if any beam diagnostic has any output on in this time step
     *
     * \param[in] output_step current step index
     * \param[in] max_step last step index
     * \param[in] output_time physical time of current step
     * \param[in] max_time physical time of last step
     */
    bool hasBeamOutput (int output_step, int max_step,
                        amrex::Real output_time, amrex::Real max_time) const
    {
        return m_output_beam_names.size() > 0 &&
            utils::doDiagnostics(m_beam_output_period, output_step, max_step,
                            output_time, max_time);
    }

    /** \brief determines if any field or beam diagnostic has any output on in this time step
     *
     * \param[in] output_step current step index
     * \param[in] max_step last step index
     * \param[in] output_time physical time of current step
     * \param[in] max_time physical time of last step
     */
    bool hasAnyOutput (int output_step, int max_step,
                       amrex::Real output_time, amrex::Real max_time) const
    {
        return hasAnyFieldOutput(output_step, max_step, output_time, max_time) ||
               hasBeamOutput(output_step, max_step, output_time, max_time);
    }

    /** \brief determines if rho is requested as a diagnostic
     */
    bool needsRho () const;

    /** \brief determines if rho for every individual plasma is requested as a diagnostic
     */
    bool needsRhoIndividual () const;

    /** \brief calculate box which possibly was trimmed in case of slice IO
     *
     * \param[in] slice_dir slicing direction
     * \param[in,out] domain_3d domain box to be possibly trimmed to a slice box
     * \param[in,out] rbox_3d real box to be possibly trimmed to a slice box
     */
    void TrimIOBox (int slice_dir, amrex::Box& domain_3d, amrex::RealBox& rbox_3d);

    /** \brief resizes the FArrayBox of the diagnostics to the currently calculated box
     *
     * \param[in] a_domain box to which the Geometry of the diagnostics will be resized to
     * \param[in] lev MR level
     * \param[in] geom geometry of the full simulation domain
     * \param[in] output_step current step index
     * \param[in] max_step last step index
     * \param[in] output_time physical time of current step
     * \param[in] max_time physical time of last step
     */
    void ResizeFDiagFAB (const amrex::Box a_domain, const int lev,
                         amrex::Geometry const& geom, int output_step, int max_step,
                         amrex::Real output_time, amrex::Real max_time);

private:
    amrex::Vector<std::string> m_output_beam_names; /**< Component names to Write to output file */
    /** Number of iterations between consecutive output dumps.
     * Default is 0, meaning no output */
    int m_beam_output_period = 0;
    amrex::Vector<FieldDiagnosticData> m_field_data; /**< individual field diag data */
    bool m_initialized = false; /**< if this object is fully initialized */
};

#endif // DIAGNOSTIC_H_
