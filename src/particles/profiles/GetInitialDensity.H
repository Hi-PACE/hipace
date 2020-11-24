#ifndef GETINTIALDENSITY_H_
#define GETINTIALDENSITY_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

/** \brief Beam profile type, currently only Gaussian or Flattop */
enum struct BeamProfileType { Gaussian, Flattop };

/** \brief Functor gets the initial density for beam particles for a given position
 */
struct GetInitialDensity
{
    /** Constructor.
     */
    GetInitialDensity ();

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
        }
        return density;
    }
    amrex::RealVect  m_position_mean {0.,0.,0.}; /* mean in case of a Gaussian density distribution. */
    amrex::RealVect  m_position_std {0.,0.,0.};  /* rms standard deviation in case of a Gaussian density distribution */
    BeamProfileType m_profile; /* beam profile type, e.g. BeamProfileType::Flattop or BeamProfileType::Gaussian*/
    amrex::Real m_density; /**< Peak density of the beam */
};

#endif // GETINTIALDENSITY_H_
