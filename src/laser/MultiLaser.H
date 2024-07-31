/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#ifndef MULTILASER_H_
#define MULTILASER_H_

#include "Laser.H"
#include "mg_solver/HpMultiGrid.H"
#include "fields/fft_poisson_solver/fft/AnyFFT.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>
#include <AMReX_GpuComplex.H>

/** \brief describes which slice with respect to the currently calculated is used */
namespace WhichLaserSlice {
    // n stands for the time step, j for the longitudinal slice.
    // n00 is time step n, nm1 is n-1 and np1 is n+1. Similar notation for slice j.
    enum slice {
        nm1j00_r,
        nm1j00_i,
        nm1jp1_r,
        nm1jp1_i,
        nm1jp2_r,
        nm1jp2_i,
        n00j00_r,
        n00j00_i,
        n00jp1_r,
        n00jp1_i,
        n00jp2_r,
        n00jp2_i,
        np1j00_r,
        np1j00_i,
        np1jp1_r,
        np1jp1_i,
        np1jp2_r,
        np1jp2_i,
        chi,
        chi_initial,
        N
    };
}

class Fields;

class MultiPlasma;

class MultiLaser
{

    using SpectralFieldLoc = amrex::BaseFab <amrex::GpuComplex<amrex::Real>>;

public:

    /** Constructor */
    explicit MultiLaser ()
    {
        ReadParameters();
    }

    void ReadParameters ();

    /** get function for the 2D slices */
    amrex::MultiFab& getSlices () {return m_slices; }

    /** get function for the 2D slices (const version) */
    const amrex::MultiFab& getSlices () const {return m_slices; }

    /** \brief Make Laser geometry
     * \param[in] field_geom_3D 3D Geometry for level 0
     */
    void MakeLaserGeometry (const amrex::Geometry& field_geom_3D);

    /** \brief Allocate laser multifab */
    void InitData ();

    /** \brief Initialize on slice of the 3D laser field.
     *
     * \param[in] islice slice index, referring to the 3D slice
     * \param[in] comp laser component to initialize
     */
    void InitSliceEnvelope (const int islice, const int comp);

    /** \brief Read in a laser from an openPMD file */
    void GetEnvelopeFromFileHelper ();

    /** \brief Read in a laser from an openPMD file */
    template<typename input_type>
    void GetEnvelopeFromFile ();

    /** \brief Shift 2D slices in zeta
     * \param[in] islice slice index
     */
    void ShiftLaserSlices (const int islice);

    /** Write Aabs into Fields MultiFab
     * \param[in] islice slice index
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] fields Field object
     * \param[in] field_geom Geometry of the problem
     */
    void UpdateLaserAabs (const int islice, const int current_N_level, Fields& fields,
                          amrex::Vector<amrex::Geometry> const& field_geom);

    /** Put Chi from the fields and initial chi into the chi component of the laser
     * \param[in] fields Field object
     * \param[in] geom_field_lev0 Geometry of the fields on MR level 0
     */
    void InterpolateChi (const Fields& fields, amrex::Geometry const& geom_field_lev0);

    /** Fill the chi_initial component of the laser using the Plasma density function
     * \param[in] multi_plasma multi plasma to get the density function, charge and mass
     */
    void SetInitialChi (const MultiPlasma& multi_plasma);

    /** Wrapper function to advance a laser slice by 1 time step.
     * \param[in] islice slice index
     * \param[in] fields Field object
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     * \param[in] geom_field_lev0 Geometry of the fields on MR level 0
     */
    void AdvanceSlice (const int islice, const Fields& fields, amrex::Real dt, int step,
                       amrex::Geometry const& geom_field_lev0);

    /** Advance a laser slice by 1 time step using a multigrid solver.
     * The complex phase of the envelope is evaluated on-axis only, but can be generalized to everywhere.
     *
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     */
    void AdvanceSliceMG (amrex::Real dt, int step);

    /** Advance a laser slice by 1 time step using a FFT solver.
     * The complex phase of the envelope is evaluated on-axis only.
     *
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     */
    void AdvanceSliceFFT (amrex::Real dt, int step);

    /** Initialize 1 longitudinal slice of the laser, and store it in n00j00 (current time step)
     * and nm1j00 (previous time step).
     *
     * \param[in] islice slice index
     * \param[in] comp laser component to initialize
     */
    void InitLaserSlice (const int islice, const int comp);

    /** Compute in-situ laser diagnostics of current slice, store in member variable
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituComputeDiags (int step, amrex::Real time, int islice,
                             int max_step, amrex::Real max_time);

    /** Dump in-situ reduced diagnostics to file.
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituWriteToFile (int step, amrex::Real time, int max_step, amrex::Real max_time);

    /** Get the central wavelength */
    amrex::Real GetLambda0 () const { return m_lambda0; }

    /** Get the geometry of the Laser Box */
    const amrex::Geometry& GetLaserGeom () const { return m_laser_geom_3D; }

    /** If the laser geometry includes this slice
     * \param[in] islice slice index
     */
    bool HasSlice (const int islice) const {
        return GetLaserGeom().Domain().smallEnd(2) <= islice &&
               islice <= GetLaserGeom().Domain().bigEnd(2);
    }

    /** If the laser is used */
    bool UseLaser () const { return m_use_laser; }

    /** If the laser is used and the laser geometry includes this slice
     * \param[in] islice slice index
     */
    bool UseLaser (const int islice) const { return m_use_laser && HasSlice(islice); }

private:

    bool m_use_laser {false}; /**< whether a laser is used or not */
    /** Laser central wavelength.
     * he central wavelength influences the solver. As long as all the lasers are on the same grid
     * (part of MultiLaser), this must be a property of MultiLaser. */
    amrex::Real m_lambda0 {0.};
    amrex::Vector<std::string> m_names {"no_laser"}; /**< name of the laser */
    int m_nlasers; /**< Number of laser pulses */
    amrex::Vector<Laser> m_all_lasers; /**< Each is a laser pulse */
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards = {-1, -1, -1};
    std::string m_solver_type = "multigrid";
    bool m_use_phase {true};
    /** 3D Laser Geometry */
    amrex::Geometry m_laser_geom_3D;
    /** xy slice BoxArray. Contains only one box */
    amrex::BoxArray m_laser_slice_ba;
    /** xy slice DistributionMapping */
    amrex::DistributionMapping m_laser_slice_dm;
    /** slice box of laser  */
    amrex::Box m_slice_box;
    /** interpolation order for laser to field and field to laser operations */
    int m_interp_order = 1;

    /** if the lasers are initialized from openPMD file */
    bool m_laser_from_file = false;
    /** full 3D laser data stored on the host */
    amrex::FArrayBox m_F_input_file;
    /** path to input openPMD file */
    std::string m_input_file_path;
    /** name of the openPMD species in the file */
    std::string m_file_envelope_name = "laserEnvelope";
    /** index of the iteration in the openPMD file */
    int m_file_num_iteration = 0;
    /** Geometry of the laser file, 'rt' or 'xyt' */
    std::string m_file_geometry = "";

    /** Array of N slices required to compute current slice */
    amrex::MultiFab m_slices;
    amrex::Real m_MG_tolerance_rel = 1.e-4;
    amrex::Real m_MG_tolerance_abs = 0.;
    int m_MG_verbose = 0;
    /** Whether to use time-averaged RHS in envelope solver. */
    bool m_MG_average_rhs = true;
    /** hpmg solver for the envelope solver */
    std::unique_ptr<hpmg::MultiGrid> m_mg;
    /** store rhs for MG solver */
    amrex::FArrayBox m_rhs_mg;
    /** store real part of acoeff of MG solver */
    amrex::FArrayBox m_mg_acoeff_real;

    /** FFTW plan for forward C2C transform to solve Complex Poisson equation */
    AnyFFT m_forward_fft;
    /** FFTW plan for backward C2C transform to solve Complex Poisson equation */
    AnyFFT m_backward_fft;
    /** work area for both FFT plans */
    amrex::Gpu::DeviceVector<char> m_fft_work_area;
    /** Complex FAB to store the solution (e.g. laser envelope on current slice) */
    SpectralFieldLoc m_sol;
    /** Complex FAB to store the RHS in position space */
    SpectralFieldLoc m_rhs;
    /** Complex FAB to store the RHS in Fourier space */
    SpectralFieldLoc m_rhs_fourier;

    // Data for in-situ diagnostics:
    /** Number of real laser properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nrp = 6;
    /** Number of real complex properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_ncp = 1;
    /** How often the insitu laser diagnostics should be computed and written
     * Default is 0, meaning no output */
    int m_insitu_period {0};
    /** All per-slice real laser properties */
    amrex::Vector<amrex::Real> m_insitu_rdata;
    /** Sum of all per-slice real laser properties */
    amrex::Vector<amrex::Real> m_insitu_sum_rdata;
    /** All per-slice complex laser properties */
    amrex::Vector<amrex::GpuComplex<amrex::Real>> m_insitu_cdata;
    /** Prefix/path for the output files */
    std::string m_insitu_file_prefix = "diags/laser_insitu";
};

#endif // MULTILASER_H_
