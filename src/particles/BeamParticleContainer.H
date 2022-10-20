/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Remi Lehe, Severin Diederichs, atmyers
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_BeamParticleContainer_H_
#define HIPACE_BeamParticleContainer_H_

#include "profiles/GetInitialDensity.H"
#include "profiles/GetInitialMomentum.H"
#include "utils/Parser.H"
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

struct BeamBins : amrex::DenseBins<amrex::ParticleTile<0, 0, BeamIdx::nattribs, 0>::ParticleType> {

    template<class...Args>
    void build (Args&&...args) {
        amrex::DenseBins<amrex::ParticleTile<0, 0, BeamIdx::nattribs, 0>::ParticleType>::build(args...);

        const auto offset_size = numBins() + 1;
        const auto offsets_gpu = offsetsPtr();
        m_offsets_cpu.resize(offset_size);
#ifdef AMREX_USE_GPU
        amrex::Gpu::dtoh_memcpy_async(m_offsets_cpu.dataPtr(), offsets_gpu,
                                      offset_size * sizeof(index_type));
        amrex::Gpu::streamSynchronize();
#else
        std::memcpy(m_offsets_cpu.dataPtr(), offsets_gpu,
                    offset_size * sizeof(index_type));
#endif
    }

    const index_type* offsetsPtrCpu () const noexcept { return m_offsets_cpu.dataPtr(); }

    amrex::Vector<index_type> m_offsets_cpu;
};

/** \brief Container for particles of 1 beam species. */
class BeamParticleContainer
    : public amrex::ParticleTile<0, 0, BeamIdx::nattribs, 0>
{
public:
    /** Constructor */
    explicit BeamParticleContainer (std::string name) :
        amrex::ParticleTile<0,0,BeamIdx::nattribs,0>(),
        m_name(name)
    {
        ReadParameters();
    }

    /** Read parameters in the input file */
    void ReadParameters ();

    /** \brief Allocate beam particle data and initialize particles with requested beam profile
     * \param[in] geom Geometry object for the whole domain
     * \return physical time at which the simulation will start
     */
    amrex::Real InitData (const amrex::Geometry& geom);

    /** Initialize a beam with a fix number of particles per cell */
    void InitBeamFixedPPC (
        const amrex::IntVect&  a_num_particles_per_cell,
        const GetInitialDensity& get_density,
        const GetInitialMomentum& get_momentum,
        const amrex::Geometry& a_geom,
        const amrex::Real     a_zmin,
        const amrex::Real     a_zmax,
        const amrex::Real     a_radius,
        const amrex::Array<amrex::Real, 3> a_position_mean,
        const amrex::Real     a_min_density,
        const amrex::Vector<int>& random_ppc);

    /** Initialize a beam with a fix number of particles, and fixed weight */
    void InitBeamFixedWeight (int num_to_add,
                              const GetInitialMomentum& get_momentum,
                              const amrex::ParserExecutor<1>& pos_mean_x,
                              const amrex::ParserExecutor<1>& pos_mean_y,
                              const amrex::Real pos_mean_z,
                              const amrex::RealVect pos_std,
                              const amrex::Real total_charge,
                              const bool do_symmetrize,
                              const bool can, const amrex::Real zmin, const amrex::Real zmax);

#ifdef HIPACE_USE_OPENPMD
    /** Checks the input file first to determine its Datatype
     * \return physical time at which the simulation will start
     */
    amrex::Real InitBeamFromFileHelper (const std::string input_file,
                                        const bool coordinates_specified,
                                        const amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz,
                                        const amrex::Geometry& geom,
                                        amrex::Real n_0,
                                        const int num_iteration,
                                        const std::string species_name,
                                        const bool species_specified);

    /** Initialize a beam from an external input file using openPMD and HDF5
     * \return physical time at which the simulation will start
     */
    template<typename input_type>
    amrex::Real InitBeamFromFile (const std::string input_file,
                                  const bool coordinates_specified,
                                  const amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz,
                                  const amrex::Geometry& geom,
                                  amrex::Real n_0,
                                  const int num_iteration,
                                  const std::string species_name,
                                  const bool species_specified);
#endif

    /** Compute reduced beam diagnostics of current slice, store in member variable
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] bins Binning object to get particles on current slice
     * \param[in] islice0 index of the leftmost slice of this box, to select the correct bin
     * \param[in] box_offset Index of first particle of the current box.
     */
    void InSituComputeDiags (int islice, const BeamBins& bins, int islice0, const int box_offset);
    /** Dump in-situ reduced diagnostics to file.
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] geom Geometry object for the whole domain
     */
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom);
    std::string get_name () const {return m_name;}
    amrex::Real m_charge; /**< charge of each particle of this species */
    amrex::Real m_mass; /**< mass of each particle of this species */
    bool m_do_z_push {true}; /**< Pushing beam particles in z direction */
    int m_n_subcycles {1}; /**< Number of sub-cycles in the beam pusher */
    int m_finest_level {0}; /**< finest level of mesh refinement that the beam interacts with */
    /** Number of particles on upstream rank (required for IO) */
    int m_num_particles_on_upstream_ranks {0};

    unsigned long long m_total_num_particles {0};

    unsigned long long get_total_num_particles () const {return m_total_num_particles;}

    amrex::Long TotalNumberOfParticles (bool only_valid=true, bool only_local=false) const;

    bool doInSitu (int step);

private:
    std::string m_name; /**< name of the species */
    amrex::Real m_zmin; /**< Min longitudinal particle position of the beam */
    amrex::Real m_zmax; /**< Max longitudinal particle position of the beam */
    amrex::Real m_radius; /**< Radius of the can beam */
    amrex::IntVect m_ppc {1, 1, 1}; /**< Number of particles per cell in each direction */
    /** Average x position of the fixed-weight beam depending on z */
    amrex::Parser m_pos_mean_x_parser;
    /** Average y position of the fixed-weight beam depending on z */
    amrex::Parser m_pos_mean_y_parser;
    /** Width of the Gaussian beam. Only used for a fixed-weight beam */
    amrex::RealVect m_position_std {0., 0., 0.};
    amrex::Real m_duz_per_uz0_dzeta {0.}; /**< relative energy spread per dzeta */
    /** injection type, fixed_width or fixed_ppc */
    std::string m_injection_type;
    int m_num_particles; /**< Number of particles for fixed-weigth Gaussian beam */
    amrex::Real m_total_charge; /**< Total beam charge for fixed-weight Gaussian beam */
    amrex::Real m_density; /**< Peak density for fixed-weight Gaussian beam */
    amrex::Real m_min_density {0.}; /**< minimum density at which beam particles are generated */
    bool m_do_symmetrize {0}; /**< Option to symmetrize the beam */
    /** Density of plasma to convert from_file beam to normalized units */
    amrex::Real m_plasma_density = 0;
    std::string m_input_file; /**< Path to bean input file */
    /** Coordinates used in input file, are converted to Hipace Coordinates x y z respectively */
    amrex::Array<std::string, AMREX_SPACEDIM> m_file_coordinates_xyz;
    int m_num_iteration {0}; /**< the iteration of the openPMD beam */
    std::string m_species_name; /**< the name of the particle species in the beam file */
    int m_nslices; /**< number of z slices of the domain */
    /** Number of real beam properties for in-situ per-slice reduced diagnostics. */
    int m_insitu_nrp {13};
    /** Number of int beam properties for in-situ per-slice reduced diagnostics. */
    int m_insitu_nip {1};
    /** Per-slice real beam properties:
     *      0,   1,     2,   3,     4,    5,      6,    7,      8,      9,     10,   11,     12
     * sum(w), [x], [x^2], [y], [y^2], [ux], [ux^2], [uy], [uy^2], [x*ux], [y*uy], [ga], [ga^2]
     * where [] means average over all particles within slice.
     * Per-slice emittance: sqrt( abs( ([x^2]-[x]^2) * ([ux^2]-[ux]^2) - ([x*ux]-[x][ux])^2 ) ).
     * Projected emittance: Same as above AFTER averaging all these quantities over slices.
     * Energy spread: sqrt([ga^2]-[ga]^2), and same as above.
     * Np: number of particles in this slice
     */
    amrex::Vector<amrex::Real> m_insitu_rdata;
    /** Per-slice int beam properties:
     *  0
     * Np
     * Np: number of particles in this slice
     */
    amrex::Vector<int> m_insitu_idata;
    /** Sum of all per-slice real beam properties */
    amrex::Vector<amrex::Real> m_insitu_sum_rdata;
    /** Sum of all per-slice int beam properties */
    amrex::Vector<int> m_insitu_sum_idata;
    /** Prefix/path for the output files */
    std::string m_insitu_file_prefix = "diags/insitu";
    /** How often the insitu beam diagnostics should be computed and written */
    int m_insitu_period {-1};
};

#endif
