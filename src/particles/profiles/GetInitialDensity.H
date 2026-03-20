/* Copyright 2020
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef GETINTIALDENSITY_H_
#define GETINTIALDENSITY_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include "utils/Parser.H"
#include "particles/particles_utils/ShapeFactors.H"

/** \brief Beam profile type, currently only Gaussian or Flattop or Parsed */
enum struct BeamProfileType { Gaussian, Flattop, Parsed };

/** \brief Functor gets the initial density for beam particles for a given position
 */
struct GetInitialDensity
{
    /** Constructor.
     */
    GetInitialDensity () = default;
    GetInitialDensity (const std::string& name, amrex::Parser& parser);

    /** \brief returns the density for a beam particle at a given position
     * \param[in] x position in x
     * \param[in] y position in y
     * \param[in] z position in z
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (const amrex::Real x, const amrex::Real y, const amrex::Real z) const
    {
        using namespace amrex::literals;
        amrex::Real density = 0._rt;
        if        (m_profile == BeamProfileType::Gaussian) {
            amrex::Real delta_x = (x - m_position_mean[0]) / m_position_std[0];
            amrex::Real delta_y = (y - m_position_mean[1]) / m_position_std[1];
            amrex::Real delta_z = (z - m_position_mean[2]) / m_position_std[2];
            density = m_density * exp( -0.5_rt*delta_x*delta_x )
                                * exp( -0.5_rt*delta_y*delta_y )
                                * exp( -0.5_rt*delta_z*delta_z );
        } else if (m_profile == BeamProfileType::Flattop) {
            density = m_density;
        } else if (m_profile == BeamProfileType::Parsed) {
            density = m_density_func(x, y, z);
        }
        return density;
    }
    amrex::RealVect  m_position_mean {0.,0.,0.}; /* mean in case of a Gaussian density distribution. */
    amrex::RealVect  m_position_std {0.,0.,0.};  /* rms standard deviation in case of a Gaussian density distribution */
    BeamProfileType m_profile; /* beam profile type, e.g. BeamProfileType::Flattop or BeamProfileType::Gaussian*/
    amrex::Real m_density; /**< Peak density of the beam */
    amrex::ParserExecutor<3> m_density_func; /**< Density function for the plasma */
};


struct PlasmaDensityAccessor {

    /**
     * @brief Describes how the profile is computed. Possible values:
     * 0: from amrex::Parser
     * 1: from file, cartesian single precision
     * 2: from file, cartesian double precision
     * 3: from file, thetaMode single precision
     * 4: from file, thetaMode double precision
     */
    int m_profile_type = 0;
    amrex::ParserExecutor<3> m_density_func;
    double * m_d_ptr = nullptr;
    float * m_f_ptr = nullptr;
    amrex::GpuArray<std::uint64_t, 3> m_strides {};
    amrex::IntVect m_bigend;
    amrex::RealVect m_pos_offset;
    amrex::RealVect m_dx_inv;
    amrex::Real m_unitSI = 1;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    amrex::Real array (int i, int j, int k) const {
        const int ii = std::min(m_bigend[0], std::max(0, i));
        const int jj = std::min(m_bigend[1], std::max(0, j));
        const int kk = std::min(m_bigend[2], std::max(0, k));

        const uint64_t offset = ii * m_strides[0] + jj * m_strides[1] + kk * m_strides[2];

        if (m_profile_type == 1 || m_profile_type == 3) {
            return static_cast<amrex::Real>(m_f_ptr[offset]);
        } else {
            return static_cast<amrex::Real>(m_d_ptr[offset]);
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    amrex::Real operator() (amrex::Real x, amrex::Real y, amrex::Real z) const {
        if (m_profile_type == 0) {
            return m_density_func(x, y, z);
        } else if (m_profile_type == 1 || m_profile_type == 2) {
            constexpr int interp_order = 2;

            const amrex::RealVect mid = (amrex::RealVect(x,y,z) - m_pos_offset) * m_dx_inv;

            amrex::Real result = 0;
            for (int iz=0; iz<=interp_order; iz++){
                auto [sz, k] = shape_factor<interp_order>(mid[2], iz);
            for (int iy=0; iy<=interp_order; iy++){
                auto [sy, j] = shape_factor<interp_order>(mid[1], iy);
            for (int ix=0; ix<=interp_order; ix++){
                auto [sx, i] = shape_factor<interp_order>(mid[0], ix);
                result += sz * sy * sx * array(i, j, k);
            }}}
            return m_unitSI * result;
        } else if (m_profile_type == 3 || m_profile_type == 4) {
            constexpr int interp_order = 2;

            const amrex::Real r = std::sqrt(x*x + y*y);
            const amrex::Real theta = std::atan2(y, x);
            const amrex::Real rmid = (r - m_pos_offset[0]) * m_dx_inv[0];
            const amrex::Real zmid = (z - m_pos_offset[1]) * m_dx_inv[1];

            amrex::Real result = 0;
            for (int iz=0; iz<=interp_order; iz++){
                auto [sz, j] = shape_factor<interp_order>(zmid, iz);
            for (int ir=0; ir<=interp_order; ir++){
                auto [sr, i] = shape_factor<interp_order>(rmid, ir);
                result += sz * sr * array(i, j, 0);
            for (int im=1; im<=m_bigend[2]/2; im++) {
                result += sz * sr * array(i, j, 2*im-1) * std::cos(im*theta);
                result += sz * sr * array(i, j, 2*im) * std::sin(im*theta);
            }}}
            return m_unitSI * result;
        }
        return 0;
    }

    void define_parser (const amrex::ParserExecutor<3>& exe);

    void define_from_file (const std::string& path, std::shared_ptr<float>& f_data,
                           std::shared_ptr<double>& d_data,
                           const std::string& density_mesh_name);
};

#endif // GETINTIALDENSITY_H_
