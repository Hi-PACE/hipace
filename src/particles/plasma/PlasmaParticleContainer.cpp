/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Severin Diederichs
 * Weiqun Zhang, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */
#include "Hipace.H"
#include "PlasmaParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/AtomicWeightTable.H"
#include "utils/DeprecatedInput.H"
#include "utils/GPUUtil.H"
#include "utils/InsituUtil.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif
#include "particles/pusher/PlasmaParticleAdvance.H"
#include "particles/pusher/BeamParticleAdvance.H"
#include "particles/particles_utils/FieldGather.H"
#include "particles/pusher/GetAndSetPosition.H"
#include <cmath>
#include <fstream>
#include <sstream>

void
PlasmaParticleContainer::ReadParameters ()
{
    PhysConst phys_const = get_phys_const();

    amrex::ParmParse pp(m_name);
    amrex::ParmParse pp_alt("plasmas");
    std::string element = "";
    amrex::Real mass_Da = 0;
    queryWithParser(pp, "element", element);
    if (element == "electron") {
        m_charge = -phys_const.q_e;
        m_mass = phys_const.m_e;
    } else if (element == "positron") {
        m_charge = phys_const.q_e;
        m_mass = phys_const.m_e;
    } else if (element == "proton") {
        m_charge = phys_const.q_e;
        m_mass = phys_const.m_p;
    } else if (element != "") {
        m_charge = phys_const.q_e;
        mass_Da = standard_atomic_weights[element];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mass_Da != 0, "Unknown Element");
    }

    queryWithParser(pp, "mass_Da", mass_Da);
    if(mass_Da != 0) {
        m_mass = phys_const.m_p * mass_Da / 1.007276466621;
    }
    queryWithParser(pp, "mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The plasma particle mass must be specified");

    bool ion_lev_specified = queryWithParser(pp, "initial_ion_level", m_init_ion_lev);
    m_can_ionize = pp.contains("ionization_product");

    queryWithParser(pp, "can_ionize", m_can_ionize);
    if(m_can_ionize) {
        m_neutralize_background = false; // change default
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_init_ion_lev >= 0,
            "The initial ion level must be specified");
    }
    queryWithParserAlt(pp, "neutralize_background", m_neutralize_background, pp_alt);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_can_ionize || !m_neutralize_background,
        "Cannot use neutralize_background when ionization is turned on");

    if(!queryWithParser(pp, "charge", m_charge)) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_charge != 0,
            "The plasma particle charge must be specified");
    }

    if(ion_lev_specified && !m_can_ionize) {
        m_charge *= m_init_ion_lev;
    }
    queryWithParser(pp, "ionization_product", m_product_name);

    std::string density_func_str = "0.";
    DeprecatedInput(m_name, "density", "density(x,y,z)");
    DeprecatedInput(m_name, "parabolic_curvature", "density(x,y,z)",
                    "The same functionality can be obtained with the parser using "
                    "density(x,y,z) = <density> * (1 + <parabolic_curvature>*(x^2 + y^2) )" );

    bool density_func_specified = queryWithParserAlt(pp, "density(x,y,z)", density_func_str, pp_alt);
    m_density_func = makeFunctionWithParser<3>(density_func_str, m_parser, {"x", "y", "z"});

    queryWithParserAlt(pp, "min_density", m_min_density, pp_alt);

    std::string density_table_file_name{};
    m_use_density_table = queryWithParser(pp, "density_table_file", density_table_file_name);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(density_func_specified && m_use_density_table),
                                     "Can only use one plasma density from either 'density(x,y,z)'"
                                     " or 'desity_table_file', not both");
    if (m_use_density_table) {
        std::ifstream file(density_table_file_name);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file.is_open(), "Unable to open 'density_table_file'");
        std::string line;
        while (std::getline(file, line)) {
            amrex::Real pos;
            std::string density;
            if (std::getline(std::stringstream(line) >> pos, density)) {
                m_density_table.emplace(pos, density);
            }
        }
        file.close();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_density_table.empty(),
                                         "Unable to get any data out of 'density_table_file'");
    }

    queryWithParserAlt(pp, "radius", m_radius, pp_alt);
    queryWithParserAlt(pp, "hollow_core_radius", m_hollow_core_radius, pp_alt);
    queryWithParserAlt(pp, "do_symmetrize", m_do_symmetrize, pp_alt);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_hollow_core_radius < m_radius,
                                     "The hollow core plasma radius must not be smaller than the "
                                     "plasma radius itself");
    queryWithParserAlt(pp, "max_qsa_weighting_factor", m_max_qsa_weighting_factor, pp_alt);
    amrex::Vector<amrex::Real> tmp_vector;
    if (queryWithParserAlt(pp, "ppc", tmp_vector, pp_alt)){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(tmp_vector.size() == AMREX_SPACEDIM-1,
        "ppc is only specified in transverse directions for plasma particles, "
        "it is 1 in the longitudinal direction z. "
        "Hence, in 3D, plasma.ppc should only contain 2 values");
        for (int i=0; i<AMREX_SPACEDIM-1; i++) m_ppc[i] = tmp_vector[i];
    }
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    if (queryWithParser(pp, "u_mean", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_mean[idim] = loc_array[idim];
        }
    }
    bool thermal_momentum_is_specified = queryWithParser(pp, "u_std", loc_array);
    bool temperature_is_specified = queryWithParser(pp, "temperature_in_ev", m_temperature_in_ev);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(temperature_is_specified && thermal_momentum_is_specified),
         "Please specify exlusively either a temperature or the thermal momentum");
    if (thermal_momentum_is_specified) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_std[idim] = loc_array[idim];
        }
    }

    if (temperature_is_specified) {
        const PhysConst phys_const_SI = make_constants_SI();
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_std[idim] = std::sqrt( (m_temperature_in_ev * phys_const_SI.q_e)
                                       /(m_mass * (phys_const_SI.m_e / phys_const.m_e) *
                                       phys_const_SI.c * phys_const_SI.c ) );
        }
    }

    queryWithParserAlt(pp, "reorder_period", m_reorder_period, pp_alt);
    amrex::Array<int, 2> idx_array
        {Hipace::m_depos_order_xy % 2, Hipace::m_depos_order_xy % 2};
    queryWithParserAlt(pp, "reorder_idx_type", idx_array, pp_alt);
    m_reorder_idx_type = amrex::IntVect(idx_array[0], idx_array[1], 0);
    queryWithParserAlt(pp, "insitu_period", m_insitu_period, pp_alt);
    queryWithParserAlt(pp, "insitu_file_prefix", m_insitu_file_prefix, pp_alt);
}

void
PlasmaParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();

    InitParticles(m_ppc, m_u_std, m_u_mean, m_radius, m_hollow_core_radius);

    if (m_insitu_period > 0) {
#ifdef HIPACE_USE_OPENPMD
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_insitu_file_prefix !=
            Hipace::GetInstance().m_openpmd_writer.m_file_prefix,
            "Must choose a different plasma insitu file prefix compared to the full diagnostics");
#endif
        // Allocate memory for in-situ diagnostics
        m_nslices = geom.Domain().length(2);
        m_insitu_rdata.resize(m_nslices*m_insitu_nrp, 0.);
        m_insitu_idata.resize(m_nslices*m_insitu_nip, 0);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
        m_insitu_sum_idata.resize(m_insitu_nip, 0);
    }
}

void
PlasmaParticleContainer::ReorderParticles (const int islice)
{
    HIPACE_PROFILE("PlasmaParticleContainer::ReorderParticles()");
    if (m_reorder_period > 0 && islice % m_reorder_period == 0) {
        SortParticlesForDeposition(m_reorder_idx_type);
    }
}

void
PlasmaParticleContainer::UpdateDensityFunction ()
{
    if (!m_use_density_table) return;
    amrex::Real c_t = get_phys_const().c * Hipace::m_physical_time;
    auto iter = m_density_table.lower_bound(c_t);
    if (iter == m_density_table.end()) --iter;
    m_density_func = makeFunctionWithParser<3>(iter->second, m_parser, {"x", "y", "z"});
}

void
PlasmaParticleContainer::TagByLevel (const int current_N_level,
                                     amrex::Vector<amrex::Geometry> const& geom3D)
{
    HIPACE_PROFILE("PlasmaParticleContainer::TagByLevel()");

    for (PlasmaParticleIterator pti(*this); pti.isValid(); ++pti)
    {
        auto& soa = pti.GetStructOfArrays();
        const amrex::Real * const AMREX_RESTRICT pos_x = soa.GetRealData(PlasmaIdx::x).data();
        const amrex::Real * const AMREX_RESTRICT pos_y = soa.GetRealData(PlasmaIdx::y).data();
        int * const AMREX_RESTRICT cpup = soa.GetIntData(PlasmaIdx::cpu).data();

        const int lev1_idx = std::min(1, current_N_level-1);
        const int lev2_idx = std::min(2, current_N_level-1);

        const amrex::Real lo_x_lev1 = geom3D[lev1_idx].ProbLo(0);
        const amrex::Real lo_x_lev2 = geom3D[lev2_idx].ProbLo(0);

        const amrex::Real hi_x_lev1 = geom3D[lev1_idx].ProbHi(0);
        const amrex::Real hi_x_lev2 = geom3D[lev2_idx].ProbHi(0);

        const amrex::Real lo_y_lev1 = geom3D[lev1_idx].ProbLo(1);
        const amrex::Real lo_y_lev2 = geom3D[lev2_idx].ProbLo(1);

        const amrex::Real hi_y_lev1 = geom3D[lev1_idx].ProbHi(1);
        const amrex::Real hi_y_lev2 = geom3D[lev2_idx].ProbHi(1);

        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (int ip) {
                const amrex::Real xp = pos_x[ip];
                const amrex::Real yp = pos_y[ip];

                if (current_N_level > 2 &&
                    lo_x_lev2 < xp && xp < hi_x_lev2 &&
                    lo_y_lev2 < yp && yp < hi_y_lev2) {
                    // level 2
                    cpup[ip] = 2;
                } else if (current_N_level > 1 &&
                    lo_x_lev1 < xp && xp < hi_x_lev1 &&
                    lo_y_lev1 < yp && yp < hi_y_lev1) {
                    // level 1
                    cpup[ip] = 1;
                } else {
                    // level 0
                    cpup[ip] = 0;
                }
            }
        );
    }
}

void
PlasmaParticleContainer::
IonizationModule (const int lev,
                  const amrex::Geometry& geom,
                  const Fields& fields,
                  const amrex::Real background_density_SI)
{
    HIPACE_PROFILE("PlasmaParticleContainer::IonizationModule()");

    using namespace amrex::literals;

    if (!m_can_ionize) return;
    const PhysConst phys_const = get_phys_const();

    // Loop over particle boxes with both ion and electron Particle Containers at the same time
    for (amrex::MFIter mfi_ion = MakeMFIter(0, DfltMfi); mfi_ion.isValid(); ++mfi_ion)
    {
        // Extract field array from FabArray
        const amrex::FArrayBox& slice_fab = fields.getSlices(lev)[mfi_ion];
        Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
        const int psi_comp = Comps[WhichSlice::This]["Psi"];
        const int ez_comp = Comps[WhichSlice::This]["Ez"];
        const int bx_comp = Comps[WhichSlice::This]["Bx"];
        const int by_comp = Comps[WhichSlice::This]["By"];
        const int bz_comp = Comps[WhichSlice::This]["Bz"];

        // Extract properties associated with physical size of the box
        const amrex::Real dx_inv = geom.InvCellSize(0);
        const amrex::Real dy_inv = geom.InvCellSize(1);

        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, geom, slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, geom, slice_fab.box());

        const int depos_order_xy = Hipace::m_depos_order_xy;

        auto& plevel_ion = GetParticles(0);
        auto index = std::make_pair(mfi_ion.index(), mfi_ion.LocalTileIndex());
        if(plevel_ion.find(index) == plevel_ion.end()) continue;
        auto& ptile_elec = m_product_pc->DefineAndReturnParticleTile(0,
            mfi_ion.index(), mfi_ion.LocalTileIndex());
        auto& ptile_ion = plevel_ion.at(index);

        auto& soa_ion = ptile_ion.GetStructOfArrays(); // For momenta and weights

        const amrex::Real clightsq = 1.0_rt / ( phys_const.c * phys_const.c );
        // calcuation of E0 in SI units for denormalization
        const amrex::Real wp = std::sqrt(static_cast<double>(background_density_SI) *
                                         PhysConstSI::q_e*PhysConstSI::q_e /
                                         (PhysConstSI::ep0 * PhysConstSI::m_e) );
        const amrex::Real E0 = Hipace::GetInstance().m_normalized_units ?
                               wp * PhysConstSI::m_e * PhysConstSI::c / PhysConstSI::q_e : 1;

        int * const ion_lev = soa_ion.GetIntData(PlasmaIdx::ion_lev).data();
        const amrex::Real * const x_prev = soa_ion.GetRealData(PlasmaIdx::x_prev).data();
        const amrex::Real * const y_prev = soa_ion.GetRealData(PlasmaIdx::y_prev).data();
        const amrex::Real * const uxp = soa_ion.GetRealData(PlasmaIdx::ux_half_step).data();
        const amrex::Real * const uyp = soa_ion.GetRealData(PlasmaIdx::uy_half_step).data();
        const amrex::Real * const psip =soa_ion.GetRealData(PlasmaIdx::psi_half_step).data();
        const int * const idp = soa_ion.GetIntData(PlasmaIdx::id).data();
        const int * const cpup = soa_ion.GetIntData(PlasmaIdx::cpu).data();

        // Make Ion Mask and load ADK prefactors
        // Ion Mask is necessary to only resize electron particle tile once
        amrex::Gpu::DeviceVector<uint8_t> ion_mask(ptile_ion.numParticles(), 0);
        uint8_t* AMREX_RESTRICT p_ion_mask = ion_mask.data();
        amrex::Gpu::DeviceScalar<uint32_t> num_new_electrons(0);
        uint32_t* AMREX_RESTRICT p_num_new_electrons = num_new_electrons.dataPtr();
        amrex::Real* AMREX_RESTRICT adk_prefactor = m_adk_prefactor.data();
        amrex::Real* AMREX_RESTRICT adk_exp_prefactor = m_adk_exp_prefactor.data();
        amrex::Real* AMREX_RESTRICT adk_power = m_adk_power.data();

        long num_ions = ptile_ion.numParticles();

        amrex::ParallelForRNG(num_ions,
            [=] AMREX_GPU_DEVICE (long ip, const amrex::RandomEngine& engine) {

            if (idp[ip] < 0 || cpup[ip] != lev) return;

            // avoid temp slice
            const amrex::Real xp = x_prev[ip];
            const amrex::Real yp = y_prev[ip];

            // define field at particle position reals
            amrex::ParticleReal ExmByp = 0., EypBxp = 0., Ezp = 0.;
            amrex::ParticleReal Bxp = 0., Byp = 0., Bzp = 0.;

            doGatherShapeN(xp, yp,
                           ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, slice_arr,
                           psi_comp, ez_comp, bx_comp, by_comp, bz_comp,
                           dx_inv, dy_inv, x_pos_offset, y_pos_offset, depos_order_xy);

            const amrex::ParticleReal Exp = ExmByp + Byp * phys_const.c;
            const amrex::ParticleReal Eyp = EypBxp - Bxp * phys_const.c;
            const amrex::ParticleReal Ep = std::sqrt( Exp*Exp + Eyp*Eyp + Ezp*Ezp )*E0;

            // Compute probability of ionization p
            const amrex::Real gammap = (1.0_rt + uxp[ip] * uxp[ip] * clightsq
                                               + uyp[ip] * uyp[ip] * clightsq
                                               + psip[ip]* psip[ip] ) / ( 2.0_rt * psip[ip] );
            const int ion_lev_loc = ion_lev[ip];
            // gamma / (psi + 1) to complete dt for QSA
            amrex::Real w_dtau = gammap / psip[ip] * adk_prefactor[ion_lev_loc] *
                std::pow(Ep, adk_power[ion_lev_loc]) *
                std::exp( adk_exp_prefactor[ion_lev_loc]/Ep );
            amrex::Real p = 1._rt - std::exp( - w_dtau );

            amrex::Real random_draw = amrex::Random(engine);
            if (random_draw < p)
            {
                ion_lev[ip] += 1;
                p_ion_mask[ip] = 1;
                amrex::Gpu::Atomic::Add( p_num_new_electrons, 1u );
            }
        });
        amrex::Gpu::streamSynchronize();

        if (num_new_electrons.dataValue() == 0) continue;

        if(Hipace::m_verbose >= 3) {
            amrex::Print() << "Number of ionized Plasma Particles: "
            << num_new_electrons.dataValue() << "\n";
        }


        // resize electron particle tile
        const auto old_size = ptile_elec.numParticles();
        const auto new_size = old_size + num_new_electrons.dataValue();
        ptile_elec.resize(new_size);

        // Load electron soa and aos after resize
        const long pid_start = ParticleType::NextID();
        ParticleType::NextID(pid_start + num_new_electrons.dataValue());

        auto arrdata_ion = ptile_ion.GetStructOfArrays().realarray();
        auto arrdata_elec = ptile_elec.GetStructOfArrays().realarray();
        auto int_arrdata_elec = ptile_elec.GetStructOfArrays().intarray();

        const int init_ion_lev = m_product_pc->m_init_ion_lev;

        amrex::Gpu::DeviceScalar<uint32_t> ip_elec(0);
        uint32_t * AMREX_RESTRICT p_ip_elec = ip_elec.dataPtr();

        amrex::ParallelFor(num_ions,
            [=] AMREX_GPU_DEVICE (long ip) {

            if(p_ion_mask[ip] != 0) {
                const long pid = amrex::Gpu::Atomic::Add( p_ip_elec, 1u );
                const long pidx = pid + old_size;

                // Copy ion data to new electron
                int_arrdata_elec[PlasmaIdx::id ][pidx] = pid_start + pid;
                int_arrdata_elec[PlasmaIdx::cpu][pidx] = lev; // current level
                arrdata_elec[PlasmaIdx::x      ][pidx] = arrdata_ion[PlasmaIdx::x     ][ip];
                arrdata_elec[PlasmaIdx::y      ][pidx] = arrdata_ion[PlasmaIdx::y     ][ip];
                arrdata_elec[PlasmaIdx::z      ][pidx] = arrdata_ion[PlasmaIdx::z     ][ip];

                arrdata_elec[PlasmaIdx::w      ][pidx] = arrdata_ion[PlasmaIdx::w     ][ip];
                arrdata_elec[PlasmaIdx::ux     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::uy     ][pidx] = 0._rt;
                // later we could consider adding a finite temperature to the ionized electrons
                arrdata_elec[PlasmaIdx::psi    ][pidx] = 1._rt;
                arrdata_elec[PlasmaIdx::x_prev ][pidx] = arrdata_ion[PlasmaIdx::x_prev][ip];
                arrdata_elec[PlasmaIdx::y_prev ][pidx] = arrdata_ion[PlasmaIdx::y_prev][ip];
                arrdata_elec[PlasmaIdx::ux_half_step ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::uy_half_step ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::psi_half_step][pidx] = 1._rt;
#ifdef HIPACE_USE_AB5_PUSH
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int iforce = PlasmaIdx::Fx1; iforce <= PlasmaIdx::Fpsi5; ++iforce) {
                    arrdata_elec[iforce][pidx] = 0._rt;
                }
#endif
                int_arrdata_elec[PlasmaIdx::ion_lev][pidx] = init_ion_lev;
            }
        });

        // synchronize before ion_mask and ip_elec go out of scope
        amrex::Gpu::streamSynchronize();
    }
}

bool
PlasmaParticleContainer::doInSitu (int step)
{
    return (m_insitu_period > 0 && step % m_insitu_period == 0);
}

void
PlasmaParticleContainer::InSituComputeDiags (int islice)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InSituComputeDiags()");

    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT(m_insitu_rdata.size()>0 && m_insitu_idata.size()>0 &&
                        m_insitu_sum_rdata.size()>0 && m_insitu_sum_idata.size()>0);

    const PhysConst phys_const = get_phys_const();
    const amrex::Real clight_inv = 1.0_rt/phys_const.c;
    const amrex::Real clightsq_inv = 1.0_rt/(phys_const.c*phys_const.c);

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(*this); pti.isValid(); ++pti)
    {
        // loading the data
        const auto ptd = pti.GetParticleTile().getParticleTileData();

        amrex::Long const num_particles = pti.numParticles();

        // Tuple contains:
        //      0,   1,     2,   3,     4,    5,      6,    7,      8,      9,     10,    11,    12,   13
        // sum(w), <x>, <x^2>, <y>, <y^2>, <ux>,   <ux^2>, <uy>, <uy^2>,  <uz>,  <uz^2>, <ga>, <ga^2>, np
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
            [=] AMREX_GPU_DEVICE (int ip) -> ReduceTuple
            {
                const amrex::Real xp   = ptd.pos(0, ip);
                const amrex::Real yp   = ptd.pos(1, ip);
                const amrex::Real uxp  = ptd.rdata(PlasmaIdx::ux )[ip] * clight_inv; // proper velocity to u
                const amrex::Real uyp  = ptd.rdata(PlasmaIdx::uy )[ip] * clight_inv;
                const amrex::Real psip = ptd.rdata(PlasmaIdx::psi)[ip];
                const amrex::Real wp   = ptd.rdata(PlasmaIdx::w  )[ip];

                if (ptd.id(ip) < 0) {
                    return{0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt,
                        0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0._rt, 0};
                }

                // particle's Lorentz factor
                const amrex::Real gamma = (1.0_rt + uxp*uxp*clightsq_inv
                                                  + uyp*uyp*clightsq_inv + psip*psip)/(2.0_rt*psip);
                amrex::Real uzp = (gamma - psip); // the *c from uz cancels with the /c from the proper velocity conversion

                return {wp,
                        wp*xp,
                        wp*xp*xp,
                        wp*yp,
                        wp*yp*yp,
                        wp*uxp,
                        wp*uxp*uxp,
                        wp*uyp,
                        wp*uyp*uyp,
                        wp*uzp,
                        wp*uzp*uzp,
                        wp*gamma,
                        wp*gamma*gamma,
                        1};
            });

        ReduceTuple a = reduce_data.value();
        const amrex::Real sum_w0 = amrex::get< 0>(a);
        const amrex::Real sum_w_inv = sum_w0 <= 0._rt ? 0._rt : 1._rt/sum_w0;

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
}

void
PlasmaParticleContainer::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InSituWriteToFile()");

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
    const int is_normalized_units = Hipace::m_normalized_units;

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
        {"is_normalized_units", &is_normalized_units},
        {"[x]"     , &m_insitu_rdata[1*nslices], nslices},
        {"[x^2]"   , &m_insitu_rdata[2*nslices], nslices},
        {"[y]"     , &m_insitu_rdata[3*nslices], nslices},
        {"[y^2]"   , &m_insitu_rdata[4*nslices], nslices},
        {"[ux]"    , &m_insitu_rdata[5*nslices], nslices},
        {"[ux^2]"  , &m_insitu_rdata[6*nslices], nslices},
        {"[uy]"    , &m_insitu_rdata[7*nslices], nslices},
        {"[uy^2]"  , &m_insitu_rdata[8*nslices], nslices},
        {"[uz]"    , &m_insitu_rdata[9*nslices], nslices},
        {"[uz^2]"  , &m_insitu_rdata[10*nslices], nslices},
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
            {"[uz]"  , &(m_insitu_sum_rdata[ 9] /= sum_w0)},
            {"[uz^2]", &(m_insitu_sum_rdata[10] /= sum_w0)},
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
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu plasma diagnostics");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu plasma diagnostics. "
        "Maybe the specified subdirectory does not exist");
#endif

    // reset arrays for insitu data
    for (auto& x : m_insitu_rdata) x = 0.;
    for (auto& x : m_insitu_idata) x = 0;
    for (auto& x : m_insitu_sum_rdata) x = 0.;
    for (auto& x : m_insitu_sum_idata) x = 0;
}
