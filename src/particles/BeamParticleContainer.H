#ifndef HIPACE_BeamParticleContainer_H_
#define HIPACE_BeamParticleContainer_H_

#include "profiles/GetInitialDensity.H"
#include "profiles/GetInitialMomentum.H"
#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

/** \brief Map names and indices for beam particles attributes (SoA data) */
struct BeamIdx
{
    enum {
        w = 0,      // weight
        ux, uy, uz, // momentum
        nattribs
    };
};

/** \brief Container for particles of 1 beam species. */
class BeamParticleContainer
    : public amrex::ParticleContainer<0, 0, BeamIdx::nattribs>
{
public:
    /** Constructor */
    explicit BeamParticleContainer (amrex::AmrCore* amr_core, std::string name) :
        amrex::ParticleContainer<0,0,BeamIdx::nattribs>(amr_core->GetParGDB()),
        m_name(name)
    {
        ReadParameters();
    }

    /** Read parameters in the input file */
    void ReadParameters ();

    /** Allocate data for the beam particles and initialize particles with requested beam profile
     */
    void InitData (const amrex::Geometry& geom);

    /** Initialize a beam with a fix number of particles per cell */
    void InitBeamFixedPPC (
        const amrex::IntVect&  a_num_particles_per_cell,
        const GetInitialDensity& get_density,
        const GetInitialMomentum& get_momentum,
        const amrex::Geometry& a_geom,
        const amrex::Real     a_zmin,
        const amrex::Real     a_zmax,
        const amrex::Real     a_radius);

    /** Initialize a beam with a fix number of particles, and fixed weight */
    void InitBeamFixedWeight (int num_to_add,
                              const GetInitialMomentum& get_momentum,
                              const amrex::RealVect pos_mean,
                              const amrex::RealVect pos_std,
                              const amrex::Real total_charge,
                              const bool do_symmetrize,
                              const amrex::Real dx_per_dzeta,
                              const amrex::Real dy_per_dzeta);

    /** Initialize a beam from an external input file using openPMD and HDF5 */
    void InitBeamFromFileHelper (std::string input_file,
                                 bool coordinates_specified,
                                 amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz);

    /** Checks the input file first to determine its Datatype*/
    template<typename input_type>
    void InitBeamFromFile (std::string input_file,
                           bool coordinates_specified,
                           amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz);

    std::string get_name () {return m_name;};
    bool m_do_z_push {true}; /**< Pushing beam particles in z direction */
private:
    std::string m_input_file; /**< path to bean input file */
    /** Coordinates used in input file, are converted to Hipace Coordinates x y z respectively */
    amrex::Array<std::string, AMREX_SPACEDIM> m_file_coordinates_xyz;
    std::string m_name; /**< name of the species */
    amrex::Real m_zmin; /**< Min longitudinal position of the can beam */
    amrex::Real m_zmax; /**< Max longitudinal position of the can beam */
    amrex::Real m_radius; /**< Radius of the can beam */
    amrex::IntVect m_ppc {1, 1, 1}; /**< Number of particles per cell in each direction */
    /** Average position of the Gaussian beam. Only used for a fixed-weight beam */
    amrex::RealVect m_position_mean {0., 0., 0.};
    /** Width of the Gaussian beam. Only used for a fixed-weight beam */
    amrex::RealVect m_position_std {0., 0., 0.};
    amrex::Real m_dx_per_dzeta {0.}; /**< tilt in x direction */
    amrex::Real m_dy_per_dzeta {0.}; /**< tilt in y direction */
    /** injection type, fixed_width or fixed_ppc */
    std::string m_injection_type;
    int m_num_particles; /**< Number of particles for fixed-weigth Gaussian beam */
    amrex::Real m_total_charge; /**< Total beam charge for fixed-weight Gaussian beam */
    amrex::Real m_density; /**< Peak density for fixed-weight Gaussian beam */
    bool m_do_symmetrize {0}; /**< Option to symmetrize the beam */
};

/** \brief Iterator over boxes in a particle container */
class BeamParticleIterator : public amrex::ParIter<0,0,BeamIdx::nattribs>
{
public:
    using amrex::ParIter<0,0,BeamIdx::nattribs>::ParIter;
    /** Constructor */
    BeamParticleIterator (ContainerType& pc, int level): ParIter(pc, level) {}
};

#endif
