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
#include "pusher/PlasmaParticleAdvance.H"
#include "pusher/BeamParticleAdvance.H"
#include "pusher/FieldGather.H"
#include "pusher/GetAndSetPosition.H"
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
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!Hipace::GetInstance().m_normalized_units,
            "Cannot use Ionization Module in normalized units");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_init_ion_lev >= 0,
            "The initial Ion level must be specified");
    }
    queryWithParserAlt(pp, "neutralize_background", m_neutralize_background, pp_alt);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_can_ionize || !m_neutralize_background,
        "Cannot use neutralize_background for Ion plasma");

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

    queryWithParser(pp, "level", m_level);
    queryWithParserAlt(pp, "radius", m_radius, pp_alt);
    queryWithParserAlt(pp, "hollow_core_radius", m_hollow_core_radius, pp_alt);
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
}

void
PlasmaParticleContainer::InitData ()
{
    reserveData();
    resizeData();

    InitParticles(m_ppc, m_u_std, m_u_mean, m_radius, m_hollow_core_radius);

    m_num_exchange = TotalNumberOfParticles();
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
PlasmaParticleContainer::
IonizationModule (const int lev,
                  const amrex::Geometry& geom,
                  const Fields& fields)
{
    HIPACE_PROFILE("PlasmaParticleContainer::IonizationModule()");

    using namespace amrex::literals;

    if (!m_can_ionize) return;
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = geom.CellSize();
    const PhysConst phys_const = make_constants_SI();

    // Loop over particle boxes with both ion and electron Particle Containers at the same time
    for (amrex::MFIter mfi_ion = MakeMFIter(lev, DfltMfi); mfi_ion.isValid(); ++mfi_ion)
    {
        // Extract field array from FabArray
        const amrex::FArrayBox& slice_fab = fields.getSlices(lev, WhichSlice::This)[mfi_ion];
        Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
        const int exmby_comp = Comps[WhichSlice::This]["ExmBy"];
        const int eypbx_comp = Comps[WhichSlice::This]["EypBx"];
        const int ez_comp = Comps[WhichSlice::This]["Ez"];
        const int bx_comp = Comps[WhichSlice::This]["Bx"];
        const int by_comp = Comps[WhichSlice::This]["By"];
        const int bz_comp = Comps[WhichSlice::This]["Bz"];

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, geom, slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, geom, slice_fab.box());

        const int depos_order_xy = Hipace::m_depos_order_xy;

        auto& plevel_ion = GetParticles(lev);
        auto index = std::make_pair(mfi_ion.index(), mfi_ion.LocalTileIndex());
        if(plevel_ion.find(index) == plevel_ion.end()) continue;
        auto& ptile_elec = m_product_pc->DefineAndReturnParticleTile(lev,
            mfi_ion.index(), mfi_ion.LocalTileIndex());
        auto& ptile_ion = plevel_ion.at(index);

        auto& soa_ion = ptile_ion.GetStructOfArrays(); // For momenta and weights
        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto getPosition = GetParticlePosition<PTileType>(ptile_ion);

        const amrex::Real clightsq = 1.0_rt / ( phys_const.c * phys_const.c );

        int * const ion_lev = soa_ion.GetIntData(PlasmaIdx::ion_lev).data();
        const amrex::Real * const uxp = soa_ion.GetRealData(PlasmaIdx::ux).data();
        const amrex::Real * const uyp = soa_ion.GetRealData(PlasmaIdx::uy).data();
        const amrex::Real * const psip = soa_ion.GetRealData(PlasmaIdx::psi).data();

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

            amrex::ParticleReal xp, yp, zp;
            int pid;
            getPosition(ip, xp, yp, zp, pid);

            if (pid < 0) return;

            // define field at particle position reals
            amrex::ParticleReal ExmByp = 0., EypBxp = 0., Ezp = 0.;
            amrex::ParticleReal Bxp = 0., Byp = 0., Bzp = 0.;

            doGatherShapeN(xp, yp,
                           ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp, slice_arr,
                           exmby_comp, eypbx_comp, ez_comp, bx_comp, by_comp, bz_comp,
                           dx_arr, x_pos_offset, y_pos_offset, depos_order_xy);

            const amrex::ParticleReal Exp = ExmByp + Byp * phys_const.c;
            const amrex::ParticleReal Eyp = EypBxp - Bxp * phys_const.c;
            const amrex::ParticleReal Ep = std::sqrt( Exp*Exp + Eyp*Eyp + Ezp*Ezp );

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
        ParticleType* pstruct_elec = ptile_elec.GetArrayOfStructs()().data();
        const int procID = amrex::ParallelDescriptor::MyProc();
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
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                pstruct_elec[pidx].id()   = pid_start + pid;
                pstruct_elec[pidx].cpu()  = procID;
                pstruct_elec[pidx].pos(0) = xp;
                pstruct_elec[pidx].pos(1) = yp;
                pstruct_elec[pidx].pos(2) = zp;

                arrdata_elec[PlasmaIdx::w       ][pidx] = arrdata_ion[PlasmaIdx::w     ][ip];
                arrdata_elec[PlasmaIdx::w0      ][pidx] = arrdata_ion[PlasmaIdx::w0    ][ip];
                arrdata_elec[PlasmaIdx::ux      ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::uy      ][pidx] = 0._rt;
                // later we could consider adding a finite temperature to the ionized electrons
                arrdata_elec[PlasmaIdx::psi     ][pidx] = 1._rt;
                arrdata_elec[PlasmaIdx::x_prev  ][pidx] = arrdata_ion[PlasmaIdx::x_prev][ip];
                arrdata_elec[PlasmaIdx::y_prev  ][pidx] = arrdata_ion[PlasmaIdx::y_prev][ip];
                arrdata_elec[PlasmaIdx::ux_temp ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::uy_temp ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::psi_temp][pidx] = 1._rt;
                arrdata_elec[PlasmaIdx::Fx1     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fx2     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fx3     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fx4     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fx5     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fy1     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fy2     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fy3     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fy4     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fy5     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fux1    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fux2    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fux3    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fux4    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fux5    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fuy1    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fuy2    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fuy3    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fuy4    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fuy5    ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fpsi1   ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fpsi2   ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fpsi3   ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fpsi4   ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::Fpsi5   ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::x0      ][pidx] = arrdata_ion[PlasmaIdx::x0    ][ip];
                arrdata_elec[PlasmaIdx::y0      ][pidx] = arrdata_ion[PlasmaIdx::y0    ][ip];
                int_arrdata_elec[PlasmaIdx::ion_lev][pidx] = init_ion_lev;
            }
        });
    }
}
