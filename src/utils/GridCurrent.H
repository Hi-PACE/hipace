#ifndef GRIDCURRENT_H_
#define GRIDCURRENT_H_

// #include "particles/MultiBeam.H"
// #include "particles/BeamParticleContainer.H"
// #include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include <AMReX_AmrCore.H>

/** \brief class handling a current directly written to the grid */
class GridCurrent
{
private:

    /** Whether to use a grid current */
    bool m_use_grid_current = false;
    /** Average position of the Gaussian grid current. */
    amrex::RealVect m_position_mean {0., 0., 0.};
    /** Width of the Gaussian grid current. */
    amrex::RealVect m_position_std {0., 0., 0.};
    amrex::Real m_amplitude {0.}; /**< peak amplitude for grid current */

public:
    /** Constructor */
    explicit GridCurrent ();

    /** calculate the adaptive time step based on the beam energy
     * \param[in,out] dt the time step
     */
    void
    DepositCurrentSlice (Fields& fields, const amrex::Geometry& geom, int const lev, const int islice);



};

#endif // GRIDCURRENT_H_
