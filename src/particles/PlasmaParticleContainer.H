#ifndef HIPACE_PlasmaParticleContainer_H_
#define HIPACE_PlasmaParticleContainer_H_

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>
#include "fields/Fields.H"

/** \brief Map names and indices for plasma particles attributes (SoA data) */
struct PlasmaIdx
{
    enum {
        w = 0,                               // weight
        w0,                                  // initial weight
        ux, uy,                              // momentum
        psi,                                 //
        x_prev, y_prev,                      // temporary position
        ux_temp, uy_temp,                    // momentum
        psi_temp,                            //
        Fx1, Fx2, Fx3, Fx4, Fx5,             //
        Fy1, Fy2, Fy3, Fy4, Fy5,             //
        Fux1, Fux2, Fux3, Fux4, Fux5,        //
        Fuy1, Fuy2, Fuy3, Fuy4, Fuy5,        //
        Fpsi1, Fpsi2, Fpsi3, Fpsi4, Fpsi5,   //
        x0, y0,                              // initial positions
        nattribs
    };
    enum {
        q_z,                                 // charge number
        int_nattribs
    };
};

/** \brief Container for particles of 1 plasma species. */
class PlasmaParticleContainer
    : public amrex::ParticleContainer<0, 0, PlasmaIdx::nattribs, PlasmaIdx::int_nattribs>
{
public:
    /** Constructor */
    explicit PlasmaParticleContainer (amrex::AmrCore* amr_core, std::string name) :
        amrex::ParticleContainer<0,0,PlasmaIdx::nattribs,PlasmaIdx::int_nattribs>(amr_core->GetParGDB()),
        m_name(name)
    {
        ReadParameters();
    }

    /** Read parameters from the input file
     */
    void ReadParameters ();

    /** Allocate data for the beam particles and initialize particles with requested beam profile
     */
    void InitData ();

    /** Initialize 1 xy slice of particles, with fixed number of particles per cell
     *
     * \param[in] a_num_particles_per_cell number of particles per cell in x and y
     * \param[in] a_thermal_momentum_std standard deviation of the momentum distribution (3d)
     * \param[in] a_thermal_momentum_mean average momentum (3d)
     * \param[in] a_density plasma density (SI)
     * \param[in] a_radius plasma radius. Only particles with x**2+y**2<a_radius**2 are injected
     */
    void InitParticles (
        const amrex::IntVect& a_num_particles_per_cell,
        const amrex::RealVect& a_thermal_momentum_std,
        const amrex::RealVect& a_thermal_momentum_mean,
        const amrex::Real a_density,
        const amrex::Real a_radius);

    void InitIonizationModule (const amrex::Geometry& geom,
                               PlasmaParticleContainer* product_pc);

    void IonizationModule (const int lev,
                           const amrex::Geometry& geom,
                           Fields& fields);

    amrex::Real m_density {0}; /**< Density of the plasma */
    /** maximum weighting factor gamma/(Psi +1) before particle is regarded as violating
     *  the quasi-static approximation and is removed */
    amrex::Real m_max_qsa_weighting_factor {35.};
    amrex::Real m_radius {std::numeric_limits<amrex::Real>::infinity()}; /**< radius of the plasma */
    /** defines the channel radius of a parabolic plasma profile */
    amrex::Real m_channel_radius {std::numeric_limits<amrex::Real>::infinity()};
    amrex::IntVect m_ppc {0,0,1}; /**< Number of particles per cell in each direction */
    amrex::RealVect m_u_mean {0,0,0}; /**< Avg momentum in each direction normalized by m*c */
    amrex::RealVect m_u_std {0,0,0}; /**< Thermal momentum in each direction normalized by m*c */
    amrex::Long m_num_exchange {0}; /** Number of particles that will be exchanged between slices*/
    /** whether to add a neutralizing background of immobile particles of opposite charge */
    bool m_neutralize_background = true;
    amrex::Real m_mass; /**< mass of each particle of this species */
    int m_charge_number; /**< initial charge number of each particle of this species */
    bool m_can_ionize = false;
    std::string m_product_name = "";
    PlasmaParticleContainer* m_product_pc = nullptr;
    amrex::Vector<amrex::Real> m_adk_prefactor;
    amrex::Vector<amrex::Real> m_adk_exp_prefactor;
    amrex::Vector<amrex::Real> m_adk_power;

private:
    std::string m_name; /**< name of the species */
};

/** \brief Iterator over boxes in a particle container */
class PlasmaParticleIterator : public amrex::ParIter<0,0,PlasmaIdx::nattribs,PlasmaIdx::int_nattribs>
{
public:
    using amrex::ParIter<0,0,PlasmaIdx::nattribs,PlasmaIdx::int_nattribs>::ParIter;
    /** Constructor */
    PlasmaParticleIterator (ContainerType& pc, int level): ParIter(pc, level) {}
};

#endif
