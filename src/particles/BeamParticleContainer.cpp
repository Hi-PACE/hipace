/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Remi Lehe
 * Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "utils/DeprecatedInput.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/InsituUtil.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif

namespace
{
    void QueryElementSetChargeMass (amrex::ParmParse& pp, amrex::Real& charge, amrex::Real& mass)
    {
        PhysConst phys_const = get_phys_const();

        std::string element = "electron";
        queryWithParser(pp, "element", element);
        if (element == "electron"){
            charge = -phys_const.q_e;
            mass = phys_const.m_e;
        } else if (element == "proton"){
            charge = phys_const.q_e;
            mass = phys_const.m_p;
        } else if (element == "positron"){
            charge = phys_const.q_e;
            mass = phys_const.m_e;
        }
    }
}


void
BeamParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    amrex::ParmParse pp_alt("beams");
    QueryElementSetChargeMass(pp, m_charge, m_mass);
    // Overwrite element's charge and mass if user specifies them explicitly
    queryWithParser(pp, "charge", m_charge);
    queryWithParser(pp, "mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The beam particle mass must not be 0");

    DeprecatedInput(m_name, "dx_per_dzeta", "position_mean = \"x_center+(z-z_center)"
        "*dx_per_dzeta\" \"y_center+(z-z_center)*dy_per_dzeta\" \"z_center\"");
    DeprecatedInput(m_name, "dy_per_dzeta", "position_mean = \"x_center+(z-z_center)"
        "*dx_per_dzeta\" \"y_center+(z-z_center)*dy_per_dzeta\" \"z_center\"");

    getWithParser(pp, "injection_type", m_injection_type);
    queryWithParser(pp, "duz_per_uz0_dzeta", m_duz_per_uz0_dzeta);
    queryWithParser(pp, "do_z_push", m_do_z_push);
    queryWithParserAlt(pp, "insitu_period", m_insitu_period, pp_alt);
    queryWithParserAlt(pp, "insitu_file_prefix", m_insitu_file_prefix, pp_alt);
    queryWithParser(pp, "n_subcycles", m_n_subcycles);
    queryWithParser(pp, "finest_level", m_finest_level);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_n_subcycles >= 1, "n_subcycles must be >= 1");
    if (m_injection_type == "fixed_ppc" || m_injection_type == "from_file"){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_duz_per_uz0_dzeta == 0.,
        "Tilted beams and correlated energy spreads are only implemented for fixed weight beams");
    }
}

amrex::Real
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    using namespace amrex::literals;
    amrex::Real ptime {0.};
    if (m_injection_type == "fixed_ppc") {

        amrex::ParmParse pp(m_name);
        amrex::Vector<amrex::Real> tmp_vector;
        if (queryWithParser(pp, "ppc", tmp_vector)){
            AMREX_ALWAYS_ASSERT(tmp_vector.size() == AMREX_SPACEDIM);
            for (int i=0; i<AMREX_SPACEDIM; i++) m_ppc[i] = tmp_vector[i];
        }
        getWithParser(pp, "zmin", m_zmin);
        getWithParser(pp, "zmax", m_zmax);
        getWithParser(pp, "radius", m_radius);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> position_mean{0., 0., 0.};
        queryWithParser(pp, "position_mean", position_mean);
        queryWithParser(pp, "min_density", m_min_density);
        m_min_density = std::abs(m_min_density);
        amrex::Vector<int> random_ppc {false, false, false};
        queryWithParser(pp, "random_ppc", random_ppc);
        const GetInitialDensity get_density(m_name);
        const GetInitialMomentum get_momentum(m_name);
        InitBeamFixedPPC(m_ppc, get_density, get_momentum, geom, m_zmin,
                         m_zmax, m_radius, position_mean, m_min_density, random_ppc);

    } else if (m_injection_type == "fixed_weight") {

        amrex::ParmParse pp(m_name);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        bool can = false;
        amrex::Real zmin = -std::numeric_limits<amrex::Real>::infinity();
        amrex::Real zmax = std::numeric_limits<amrex::Real>::infinity();
        std::string profile = "gaussian";
        queryWithParser(pp, "profile", profile);
        if (profile == "can") {
            can = true;
            getWithParser(pp, "zmin", zmin);
            getWithParser(pp, "zmax", zmax);
        } else if (profile == "gaussian") {
            queryWithParser(pp, "zmin", zmin);
            queryWithParser(pp, "zmax", zmax);
        } else {
            amrex::Abort("Only gaussian and can are supported with fixed_weight beam injection");
        }

        std::array<std::string, 3> pos_mean_arr{"","",""};
        getWithParser(pp, "position_mean", pos_mean_arr);
        auto pos_mean_x = makeFunctionWithParser<1>(pos_mean_arr[0], m_pos_mean_x_parser, {"z"});
        auto pos_mean_y = makeFunctionWithParser<1>(pos_mean_arr[1], m_pos_mean_y_parser, {"z"});
        amrex::Real pos_mean_z = 0;
        Parser::fillWithParser(pos_mean_arr[2], pos_mean_z);

        getWithParser(pp, "position_std", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) m_position_std[idim] = loc_array[idim];
        getWithParser(pp, "num_particles", m_num_particles);
        bool charge_is_specified = queryWithParser(pp, "total_charge", m_total_charge);
        bool peak_density_is_specified = queryWithParser(pp, "density", m_density);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( charge_is_specified + peak_density_is_specified == 1,
            "Please specify exlusively either total_charge or density of the beam");
        queryWithParser(pp, "do_symmetrize", m_do_symmetrize);
        if (m_do_symmetrize) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_num_particles%4 == 0,
            "To symmetrize the beam, please specify a beam particle number divisible by 4.");

        if (peak_density_is_specified)
        {
            m_total_charge = m_density*m_charge;
            for (int idim=0; idim<AMREX_SPACEDIM; ++idim)
            {
                m_total_charge *= m_position_std[idim] * sqrt(2. * MathConst::pi);
            }
        }
        if (Hipace::m_normalized_units)
        {
            auto dx = geom.CellSizeArray();
            m_total_charge /= dx[0]*dx[1]*dx[2];
        }

        const GetInitialMomentum get_momentum(m_name);
        InitBeamFixedWeight(m_num_particles, get_momentum, pos_mean_x, pos_mean_y, pos_mean_z,
                            m_position_std, m_total_charge, m_do_symmetrize, can, zmin, zmax);

    } else if (m_injection_type == "from_file") {
#ifdef HIPACE_USE_OPENPMD
        amrex::ParmParse pp(m_name);
        amrex::ParmParse pp_alt("beams");
        getWithParser(pp, "input_file", m_input_file);
        bool coordinates_specified = queryWithParserAlt(pp, "file_coordinates_xyz",
                                                        m_file_coordinates_xyz, pp_alt);
        queryWithParserAlt(pp, "plasma_density", m_plasma_density, pp_alt);
        queryWithParserAlt(pp, "iteration", m_num_iteration, pp_alt);
        bool species_specified = queryWithParser(pp, "openPMD_species_name", m_species_name);
        if(!species_specified) {
            m_species_name = m_name;
        }

        ptime = InitBeamFromFileHelper(m_input_file, coordinates_specified, m_file_coordinates_xyz,
                                       geom, m_plasma_density, m_num_iteration, m_species_name,
                                       species_specified);
#else
        amrex::Abort("beam particle injection via external_file requires openPMD support: "
                     "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif  // HIPACE_USE_OPENPMD
    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc, fixed_weight or from_file\n");

    }

    if (m_insitu_period > 0) {
        // Allocate memory for in-situ diagnostics
        m_nslices = geom.Domain().length(2);
        m_insitu_rdata.resize(m_nslices*m_insitu_nrp, 0.);
        m_insitu_idata.resize(m_nslices*m_insitu_nip, 0);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
        m_insitu_sum_idata.resize(m_insitu_nip, 0);
    }
    /* setting total number of particles, which is required for openPMD I/O */
    m_total_num_particles = TotalNumberOfParticles();

    return ptime;
}

amrex::Long BeamParticleContainer::TotalNumberOfParticles (bool only_valid, bool only_local) const
{
    amrex::Long nparticles = 0;
    if (only_valid) {
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<unsigned long long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        auto const& ptaos = this->GetArrayOfStructs();
        ParticleType const* pp = ptaos().data();

        reduce_op.eval(ptaos.numParticles(), reduce_data,
                       [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
                       {
                           return (pp[i].id() > 0) ? 1 : 0;
                       });
        nparticles = static_cast<amrex::Long>(amrex::get<0>(reduce_data.value()));
    }
    else {
        nparticles = this->numParticles();
    }

    if (!only_local) {
        amrex::ParallelAllReduce::Sum(nparticles, amrex::ParallelContext::CommunicatorSub());
    }

    return nparticles;
}

bool
BeamParticleContainer::doInSitu (int step)
{
    if (m_insitu_period <= 0) return false;
    return step % m_insitu_period == 0;
}

void
BeamParticleContainer::InSituComputeDiags (int islice, const BeamBins& bins, int islice0,
                                           const int box_offset)
{
    HIPACE_PROFILE("BeamParticleContainer::InSituComputeDiags");

    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT(m_insitu_rdata.size()>0 && m_insitu_idata.size()>0 &&
                        m_insitu_sum_rdata.size()>0 && m_insitu_sum_idata.size()>0);

    PhysConst const phys_const = get_phys_const();
    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

    auto const& ptaos = this->GetArrayOfStructs();
    const auto& pos_structs = ptaos.begin() + box_offset;
    auto const& soa = this->GetStructOfArrays();
    const auto  wp = soa.GetRealData(BeamIdx::w).data() + box_offset;
    const auto uxp = soa.GetRealData(BeamIdx::ux).data() + box_offset;
    const auto uyp = soa.GetRealData(BeamIdx::uy).data() + box_offset;
    const auto uzp = soa.GetRealData(BeamIdx::uz).data() + box_offset;

    BeamBins::index_type const * const indices = bins.permutationPtr();
    BeamBins::index_type const * const offsets = bins.offsetsPtrCpu();
    BeamBins::index_type const cell_start = offsets[islice-islice0];
    BeamBins::index_type const cell_stop = offsets[islice-islice0+1];
    int const num_particles = cell_stop-cell_start;

    // Tuple contains:
    //      0,   1,     2,   3,     4,    5,      6,    7,      8,      9,     10,   11,     12, 13
    // sum(w), <x>, <x^2>, <y>, <y^2>, <ux>, <ux^2>, <uy>, <uy^2>, <x*ux>, <y*uy>, <ga>, <ga^2>, np
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_op.eval(
        num_particles, reduce_data,
        [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
        {
            const int ip = indices[cell_start+i];
            if (pos_structs[ip].id() < 0) {
                return{0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt,
                    0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0};
            }
            const amrex::Real gamma = std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                       + uyp[ip]*uyp[ip]*clightsq
                                                       + uzp[ip]*uzp[ip]*clightsq);
            return {wp[ip],
                    wp[ip]*pos_structs[ip].pos(0),
                    wp[ip]*pos_structs[ip].pos(0)*pos_structs[ip].pos(0),
                    wp[ip]*pos_structs[ip].pos(1),
                    wp[ip]*pos_structs[ip].pos(1)*pos_structs[ip].pos(1),
                    wp[ip]*uxp[ip],
                    wp[ip]*uxp[ip]*uxp[ip],
                    wp[ip]*uyp[ip],
                    wp[ip]*uyp[ip]*uyp[ip],
                    wp[ip]*pos_structs[ip].pos(0)*uxp[ip],
                    wp[ip]*pos_structs[ip].pos(1)*uyp[ip],
                    wp[ip]*gamma,
                    wp[ip]*gamma*gamma,
                    1};
        });

    ReduceTuple a = reduce_data.value();
    const amrex::Real sum_w0 = amrex::get< 0>(a);
    const amrex::Real sum_w_inv = sum_w0<std::numeric_limits<amrex::Real>::epsilon() ? 0._rt : 1._rt/sum_w0;

    m_insitu_rdata[islice             ] = sum_w0;
    m_insitu_rdata[islice+ 1*m_nslices] = amrex::get< 1>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 2*m_nslices] = amrex::get< 2>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 3*m_nslices] = amrex::get< 3>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 4*m_nslices] = amrex::get< 4>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 5*m_nslices] = amrex::get< 5>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 6*m_nslices] = amrex::get< 6>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 7*m_nslices] = amrex::get< 7>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 8*m_nslices] = amrex::get< 8>(a)*sum_w_inv;
    m_insitu_rdata[islice+ 9*m_nslices] = amrex::get< 9>(a)*sum_w_inv;
    m_insitu_rdata[islice+10*m_nslices] = amrex::get<10>(a)*sum_w_inv;
    m_insitu_rdata[islice+11*m_nslices] = amrex::get<11>(a)*sum_w_inv;
    m_insitu_rdata[islice+12*m_nslices] = amrex::get<12>(a)*sum_w_inv;
    m_insitu_idata[islice             ] = amrex::get<13>(a);

    m_insitu_sum_rdata[ 0] += sum_w0;
    m_insitu_sum_rdata[ 1] += amrex::get< 1>(a);
    m_insitu_sum_rdata[ 2] += amrex::get< 2>(a);
    m_insitu_sum_rdata[ 3] += amrex::get< 3>(a);
    m_insitu_sum_rdata[ 4] += amrex::get< 4>(a);
    m_insitu_sum_rdata[ 5] += amrex::get< 5>(a);
    m_insitu_sum_rdata[ 6] += amrex::get< 6>(a);
    m_insitu_sum_rdata[ 7] += amrex::get< 7>(a);
    m_insitu_sum_rdata[ 8] += amrex::get< 8>(a);
    m_insitu_sum_rdata[ 9] += amrex::get< 9>(a);
    m_insitu_sum_rdata[10] += amrex::get<10>(a);
    m_insitu_sum_rdata[11] += amrex::get<11>(a);
    m_insitu_sum_rdata[12] += amrex::get<12>(a);
    m_insitu_sum_idata[ 0] += amrex::get<13>(a);
}

void
BeamParticleContainer::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("BeamParticleContainer::InSituWriteToFile");

#ifdef HIPACE_USE_OPENPMD
    // create subdirectory
    openPMD::auxiliary::create_directories(m_insitu_file_prefix);
#endif

    // zero pad the rank number;
    std::string::size_type n_zeros = 4;
    std::string rank_num = std::to_string(amrex::ParallelDescriptor::MyProc());
    std::string pad_rank_num = std::string(n_zeros-std::min(rank_num.size(), n_zeros),'0')+rank_num;

    // open file
    std::ofstream ofs{m_insitu_file_prefix + "/reduced_" + m_name + "." + pad_rank_num + ".txt",
        std::ofstream::out | std::ofstream::app | std::ofstream::binary};

    const amrex::Real sum_w0 = m_insitu_sum_rdata[0];
    const std::size_t nslices = static_cast<std::size_t>(m_nslices);
    const amrex::Real normalized_density_factor = Hipace::m_normalized_units ?
        geom.CellSizeArray().product() : 1; // dx * dy * dz in normalized units, 1 otherwise

    // specify the structure of the data later available in python
    // avoid pointers to temporary objects as second argument, stack variables are ok
    const amrex::Vector<insitu_utils::DataNode> all_data{
        {"time"    , &time},
        {"step"    , &step},
        {"n_slices", &m_nslices},
        {"charge"  , &m_charge},
        {"mass"    , &m_mass},
        {"z_lo"    , &geom.ProbLo()[2]},
        {"z_hi"    , &geom.ProbHi()[2]},
        {"normalized_density_factor", &normalized_density_factor},
        {"[x]"     , &m_insitu_rdata[1*nslices], nslices},
        {"[x^2]"   , &m_insitu_rdata[2*nslices], nslices},
        {"[y]"     , &m_insitu_rdata[3*nslices], nslices},
        {"[y^2]"   , &m_insitu_rdata[4*nslices], nslices},
        {"[ux]"    , &m_insitu_rdata[5*nslices], nslices},
        {"[ux^2]"  , &m_insitu_rdata[6*nslices], nslices},
        {"[uy]"    , &m_insitu_rdata[7*nslices], nslices},
        {"[uy^2]"  , &m_insitu_rdata[8*nslices], nslices},
        {"[x*ux]"  , &m_insitu_rdata[9*nslices], nslices},
        {"[y*uy]"  , &m_insitu_rdata[10*nslices], nslices},
        {"[ga]"    , &m_insitu_rdata[11*nslices], nslices},
        {"[ga^2]"  , &m_insitu_rdata[12*nslices], nslices},
        {"sum(w)"  , &m_insitu_rdata[0], nslices},
        {"Np"      , &m_insitu_idata[0], nslices},
        {"average" , {
            {"[x]"   , &(m_insitu_sum_rdata[ 1] /= sum_w0)},
            {"[x^2]" , &(m_insitu_sum_rdata[ 2] /= sum_w0)},
            {"[y]"   , &(m_insitu_sum_rdata[ 3] /= sum_w0)},
            {"[y^2]" , &(m_insitu_sum_rdata[ 4] /= sum_w0)},
            {"[ux]"  , &(m_insitu_sum_rdata[ 5] /= sum_w0)},
            {"[ux^2]", &(m_insitu_sum_rdata[ 6] /= sum_w0)},
            {"[uy]"  , &(m_insitu_sum_rdata[ 7] /= sum_w0)},
            {"[uy^2]", &(m_insitu_sum_rdata[ 8] /= sum_w0)},
            {"[x*ux]", &(m_insitu_sum_rdata[ 9] /= sum_w0)},
            {"[y*uy]", &(m_insitu_sum_rdata[10] /= sum_w0)},
            {"[ga]"  , &(m_insitu_sum_rdata[11] /= sum_w0)},
            {"[ga^2]", &(m_insitu_sum_rdata[12] /= sum_w0)}
        }},
        {"total"   , {
            {"sum(w)", &m_insitu_sum_rdata[0]},
            {"Np"    , &m_insitu_sum_idata[0]}
        }}
    };

    if (ofs.tellp() == 0) {
        // write JSON header containing a NumPy structured datatype
        insitu_utils::write_header(all_data, ofs);
    }

    // write binary data according to datatype in header
    insitu_utils::write_data(all_data, ofs);

    // close file
    ofs.close();
    // assert no file errors
#ifdef HIPACE_USE_OPENPMD
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu beam diagnostics");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu beam diagnostics. "
        "Maybe the specified subdirectory does not exist");
#endif

    // reset arrays for insitu data
    for (auto& x : m_insitu_rdata) x = 0.;
    for (auto& x : m_insitu_idata) x = 0;
    for (auto& x : m_insitu_sum_rdata) x = 0.;
    for (auto& x : m_insitu_sum_idata) x = 0;
}
