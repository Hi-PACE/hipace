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

#include "particles/profiles/GetInitialDensity.H"
#include "particles/profiles/GetInitialMomentum.H"
#include "utils/Parser.H"
#include "particles/sorting/BoxSort.H"
#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

class AdaptiveTimeStep;

/** \brief Map names and indices for beam particles attributes (SoA data) */
struct BeamIdx
{
    enum {
        x=0, y, z,  // position
        w,          // weight
        ux, uy, uz, // momentum
        real_nattribs_in_buffer,
        real_nattribs=real_nattribs_in_buffer
    };
    enum {
        // no extra components stored in MultiBuffer, besides 64bit idcpu
        int_nattribs_in_buffer,
        // nsubcycles: by how many subcycles was this particle pushed already
        // nsubcycles is not stored or communicated in MultiBuffer
        nsubcycles=int_nattribs_in_buffer,
        int_nattribs
    };
};

struct WhichBeamSlice {
    enum beam_slice : int { Next=0, This, N };
};

using BeamTile = amrex::ParticleTile<
        amrex::SoAParticle<
            BeamIdx::real_nattribs,
            BeamIdx::int_nattribs
        >,
        BeamIdx::real_nattribs,
        BeamIdx::int_nattribs
    >;

using BeamTileInit = amrex::ParticleTile<
        amrex::SoAParticle<
            BeamIdx::real_nattribs_in_buffer,
            BeamIdx::int_nattribs_in_buffer
        >,
        BeamIdx::real_nattribs_in_buffer,
        BeamIdx::int_nattribs_in_buffer,
        // use PolymorphicArenaAllocator to either use Pinned or Device memory at runtime
        amrex::PolymorphicArenaAllocator
    >;

/** \brief Container for particles of 1 beam species. */
class BeamParticleContainer
{
public:
    /** Constructor */
    explicit BeamParticleContainer (std::string name) :
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

    /** Initialize a beam with a fixed number of particles per cell */
    void InitBeamFixedPPC3D ();
    void InitBeamFixedPPCSlice (const int islice, const int which_beam_slice);

    /** Initialize a beam with a fixed number of particles, and fixed weight */
    void InitBeamFixedWeight3D ();
    void InitBeamFixedWeightSlice (const int islice, const int which_beam_slice);

    /** Initialize a beam with a fixed number of particles,
     * and fixed weight using a probability density function (PDF) */
    void InitBeamFixedWeightPDF3D ();
    void InitBeamFixedWeightPDFSlice (int slice, int which_slice);

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
     */
    void InSituComputeDiags (int islice);

    /** Dump in-situ reduced diagnostics to file.
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] geom Geometry object for the whole domain
     */
    void InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom);

    /** \brief Store the finest level of every beam particle on which_slice in the cpu() attribute.
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] geom3D Geometry object for the whole domain
     * \param[in] which_slice slice to mark
     */
    void TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                     const int which_slice);

    /** \brief Reorder beam particles to speed-up push and current deposition
     * \param[in] beam_slice beam slice to reorder, WhichBeamSlice::This WhichBeamSlice::Next
     * \param[in] step current time step
     * \param[in] slice_geom Geometry object for one xy slice
     */
    void ReorderParticles (int beam_slice, int step, amrex::Geometry& slice_geom);

    /** Returns elementary charge q_e (or -q_e for electrons). */
    amrex::Real GetCharge () const {return m_charge;}

    /** Returns mass of physical species */
    amrex::Real GetMass () const {return m_mass;}

    std::string get_name () const {return m_name;}

    void shiftBeamSlices () {
        m_slice_permutation = (m_slice_permutation + 1) % WhichBeamSlice::N;
        resize(WhichBeamSlice::Next, 0, 0);
    }

    BeamTile& getBeamSlice (int which_slice) {
        return m_slices[(which_slice + m_slice_permutation) % WhichBeamSlice::N];
    }

    const BeamTile& getBeamSlice (int which_slice) const {
        return m_slices[(which_slice + m_slice_permutation) % WhichBeamSlice::N];
    }

    int getNumParticles (int which_slice) const {
        return m_num_particles_without_slipped[(which_slice + m_slice_permutation) % WhichBeamSlice::N];
    }

    int getNumParticlesIncludingSlipped (int which_slice) const {
        return m_num_particles_with_slipped[(which_slice + m_slice_permutation) % WhichBeamSlice::N];
    }

    void resize (int which_slice, int num_particles, int num_slipped_particles);

    BeamTileInit& getBeamInitSlice () {
        return m_init_slice;
    }

    void intializeSlice(int slice, int which_slice);

    uint64_t getTotalNumParticles () const {
        return m_total_num_particles;
    }

    int numRealComponents () const { return BeamIdx::real_nattribs; }
    int numIntComponents () const { return BeamIdx::int_nattribs; }

    bool communicateIdCpuComponent () const { return true; }
    bool communicateRealComponent (int rcomp) const { return rcomp < BeamIdx::real_nattribs_in_buffer; }
    bool communicateIntComponent (int icomp) const { return icomp < BeamIdx::int_nattribs_in_buffer; }

private:
    int m_slice_permutation = 0;
    std::array<BeamTile, WhichBeamSlice::N> m_slices {};
    std::array<int, WhichBeamSlice::N> m_num_particles_without_slipped {};
    std::array<int, WhichBeamSlice::N> m_num_particles_with_slipped {};
    BeamTileInit m_init_slice {};
    BoxSorter m_init_sorter;
    uint64_t m_total_num_particles = 0;
public:
    amrex::Real m_charge; /**< charge of each particle of this species */
    amrex::Real m_mass; /**< mass of each particle of this species */
    bool m_do_z_push {true}; /**< Pushing beam particles in z direction */
    int m_n_subcycles {10}; /**< Number of sub-cycles in the beam pusher */
    bool m_do_radiation_reaction {false}; /**< whether to calculate radiation losses */
    /** Number of particles on upstream rank (required for IO) */
    bool m_do_salame = false; /**< Whether this beam uses salame */
    /** Whether to reset the global ID incrementor to 1 before initializing this beam */
    bool m_do_reset_id_init {false};
    bool m_initialize_on_cpu {false};
    /** How often the insitu beam diagnostics should be computed and written
     * Default is 0, meaning no output */
    int m_insitu_period {0};
    /** Whether external fields should be used for this beam */
    bool m_use_external_fields = false;
    /** External field functions for Ex Ey Ez Bx By Bz */
    amrex::GpuArray<amrex::ParserExecutor<4>, 6> m_external_fields;
    /** Owns data for m_external_fields */
    amrex::Array<amrex::Parser, 6> m_external_fields_parser;
    /** If spin tracking is enabled for this beam */
    bool m_do_spin_tracking = false;
private:
    std::string m_name; /**< name of the species */
    /** injection type, fixed_width or fixed_ppc */
    std::string m_injection_type;
    uint64_t m_id64 = 1; /**< 64 bit ID to initialize many particles without overflowing */
    /** Min longitudinal particle position of the beam */
    amrex::Real m_zmin = -std::numeric_limits<amrex::Real>::infinity();
    /** Max longitudinal particle position of the beam */
    amrex::Real m_zmax = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_radius {std::numeric_limits<amrex::Real>::infinity()}; /**< Radius of the beam */
    /** radius of the beam insitu diagnostics */
    amrex::Real m_insitu_radius {std::numeric_limits<amrex::Real>::infinity()};
    GetInitialMomentum m_get_momentum {}; /**< momentum profile of the beam */
    /** After how many slices the particles are reordered. 0: off */
    int m_reorder_period = 0;
    /** 2D reordering index type. 0: cell, 1: node, 2: both */
    amrex::IntVect m_reorder_idx_type = {0, 0, 0};

    // fixed_ppc:

    amrex::IntVect m_ppc {1, 1, 1}; /**< Number of particles per cell in each direction */
    amrex::RealVect m_position_mean {0., 0., 0.}; /**< mean position of the beam */
    amrex::Real m_min_density {0.}; /**< minimum density at which beam particles are generated */
    amrex::IntVect m_random_ppc {0, 0, 0}; /**< if the cell position is random in each direction */
    GetInitialDensity m_get_density {}; /**< density profile of the beam */
    /** Density parser for fixed-ppc beam. Owns data for m_density_func */
    amrex::Parser m_density_parser;

    // fixed_weight:

    bool m_can_profile = false;
    /** Average x position of the fixed-weight beam depending on z */
    amrex::Parser m_pos_mean_x_parser;
    /** Average x position of the fixed-weight beam depending on z */
    amrex::ParserExecutor<1> m_pos_mean_x_func;
    /** Average y position of the fixed-weight beam depending on z */
    amrex::Parser m_pos_mean_y_parser;
    /* Average y position of the fixed-weight beam depending on z */
    amrex::ParserExecutor<1> m_pos_mean_y_func;
    /* Average z position of the fixed-weight beam depending on z */
    amrex::Real m_pos_mean_z = 0;
    /** Width of the Gaussian beam. Only used for a fixed-weight beam */
    amrex::RealVect m_position_std {0., 0., 0.};
    /** Distance at which the beam is focused, starting from its initial position */
    amrex::Real m_z_foc {0.};
    amrex::Real m_duz_per_uz0_dzeta {0.}; /**< relative energy spread per dzeta */
    amrex::Long m_num_particles; /**< Number of particles for fixed-weight Gaussian beam */
    amrex::Real m_total_charge; /**< Total beam charge for fixed-weight Gaussian beam */
    amrex::Real m_density; /**< Peak density for fixed-weight Gaussian beam */
    bool m_do_symmetrize {0}; /**< Option to symmetrize the beam */
    /** Array for the z position of all beam particles */
    amrex::PODVector<amrex::Real, amrex::PolymorphicArenaAllocator<amrex::Real>> m_z_array {};

    // fixed_weight_pdf:

    bool m_peak_density_is_specified = false; /**< if the peak density is specified */
    int m_pdf_ref_ratio = 4; /**< number of subcycles per slice for the pdf evaluation */
    amrex::Real m_total_weight = 0; /**< sum of the weights of all particles */
    amrex::ParserExecutor<1> m_pdf_func; /**< probability density function */
    /** number of particles that need to be initialized per slice */
    amrex::Vector<unsigned int> m_num_particles_slice;
    /** functions for x_mean, y_mean, x_std, y_std */
    amrex::Array<amrex::ParserExecutor<1>, 4> m_pdf_pos_func;
    /** functions for ux_mean, uy_mean, uz_mean, ux_std, uy_std, uz_std */
    amrex::Array<amrex::ParserExecutor<1>, 6> m_pdf_u_func;
    /** Owns data for all 11 Parser functions of fixed_weight_pdf */
    amrex::Vector<amrex::Parser> m_pdf_parsers;

    // from_file:

    /** Density of plasma to convert from_file beam to normalized units */
    amrex::Real m_plasma_density = 0;
    std::string m_input_file; /**< Path to bean input file */
    /** Coordinates used in input file, are converted to Hipace Coordinates x y z respectively */
    amrex::Array<std::string, AMREX_SPACEDIM> m_file_coordinates_xyz;
    int m_num_iteration {0}; /**< the iteration of the openPMD beam */
    std::string m_species_name; /**< the name of the particle species in the beam file */

    // insitu:

    int m_nslices; /**< number of z slices of the domain */
    /** Number of real beam properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nrp = 22;
    /** Number of int beam properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nip = 1;
    /** Per-slice real beam properties */
    amrex::Vector<amrex::Real> m_insitu_rdata;
    /** Per-slice int beam properties */
    amrex::Vector<int> m_insitu_idata;
    /** Sum of all per-slice real beam properties */
    amrex::Vector<amrex::Real> m_insitu_sum_rdata;
    /** Sum of all per-slice int beam properties */
    amrex::Vector<int> m_insitu_sum_idata;
    /** Prefix/path for the output files */
    std::string m_insitu_file_prefix = "diags/insitu";

    // to estimate min uz
    friend AdaptiveTimeStep;
};

#endif
