/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#ifndef LASER_H_
#define LASER_H_

#include <AMReX_Vector.H>
#include <AMReX_RealVect.H>
#include "utils/Constants.H"
#include "utils/Parser.H"
#include <AMReX_MultiFab.H>

#include <complex>

class Laser
{
public:
    /** \brief Read in a laser from an openPMD file
     *\param[in] laser_geom_3D 3D laser geometry for level 0
     */
    void GetEnvelopeFromFile (amrex::Geometry laser_geom_3D);

    Laser (std::string name);

    /** Read parameters from the input file */
    void ReadParameters (const amrex::Geometry& laser_geom_3D);

    std::string m_name {""};
    /** The way to initialize a laser (from_file/gaussian/parser)*/
    std::string m_laser_init_type = "gaussian";
    /** central wavelength of the laser */
    amrex::Real m_init_lambda0 {0.};

    // gaussian

    amrex::Real m_a0 {0.}; /**< Laser peak normalized amplitude */
    amrex::Real m_w0 {0.}; /**< Laser waist */
    amrex::Real m_CEP {0.}; /**< Laser carrier-envelope phase (CEP) */
    /** Propagation angle of the pulse in the yz plane (0 is along the z axis) */
    amrex::Real m_propagation_angle_yz {0.};
    amrex::Real m_L0 {0.}; /**< Laser length (HW 1/e in amplitude) */
    amrex::Real m_tau {0.}; /**< Laser duration (HW 1/e in amplitude) */
    amrex::Real m_STC_theta_xy {0.};/**< Initial propagation angel on YZ plane */
    amrex::Real m_zeta {0.};/**< The value of first order spatial chirp in m * s */
    amrex::Real m_beta {0.};/**< The value of first order angular chirp in  s */
    amrex::Real m_phi2 {0.};/**< The value of group delayed dispersion in s ^ -2 */
    /** Focal distance of the laser pulse */
    amrex::Real m_focal_distance {0.};
    /** Average position of the Gaussian laser pulse */
    amrex::RealVect m_position_mean {0., 0., 0.};

    // parser

    /** Custom profile*/
    amrex::Parser m_parser_lr;/**< owns data for profile_real_str */
    amrex::Parser m_parser_li;/**< owns data for profile_imag_str */
    /** stores the output function of makeFunctionWithParser for profile_real_str */
    amrex::ParserExecutor<3> m_profile_real;
    /** stores the output function of makeFunctionWithParser for profile_imag_str */
    amrex::ParserExecutor<3> m_profile_imag;

    // from_file

    /** full fp32 laser data from the openPMD file */
    std::shared_ptr<std::complex<float>> m_cf_laser_data;
    /** full fp64 laser data from the openPMD file */
    std::shared_ptr<std::complex<double>> m_cd_laser_data;
    amrex::GpuComplex<float>* m_cf_ptr = nullptr; /**< pointer to fp32 laser data */
    amrex::GpuComplex<double>* m_cd_ptr = nullptr; /**< pointer to fp64 laser data */
    amrex::GpuArray<std::uint64_t, 2> m_strides {}; /**< strides of laser data */
    amrex::IntVect m_bigend; /**< upper bound laser data */
    amrex::RealVect m_pos_offset; /**< position offset */
    amrex::RealVect m_dx_inv; /**< inverse cell size */
    amrex::Real m_unitSI = 1; /**< SI conversion factor */
    /** path to input openPMD file */
    std::string m_input_file_path;
    /** name of the openPMD species in the file */
    std::string m_file_envelope_name = "laserEnvelope";
    /** index of the iteration in the openPMD file */
    int m_file_num_iteration = 0;
    /** Geometry of the laser file, 'rt', 'xyt' or 'xyz' */
    std::string m_file_geometry = "";
};

#endif // LASER_H_
