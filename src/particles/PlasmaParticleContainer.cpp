#include "Hipace.H"
#include "PlasmaParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/AtomicWeightTable.H"
#include "pusher/PlasmaParticleAdvance.H"
#include "pusher/BeamParticleAdvance.H"
#include "pusher/FieldGather.H"
#include "pusher/GetAndSetPosition.H"
#include <cmath>

void
PlasmaParticleContainer::ReadParameters ()
{
    // normalized_units is directly queried here so we can defined the appropriate PhysConst
    // locally. We cannot use Hipace::m_phys_const as it has not been initialized when the
    // PlasmaParticleContainer constructor is called.
    amrex::ParmParse pph("hipace");
    bool normalized_units = false;
    pph.query("normalized_units", normalized_units);
    PhysConst phys_const = normalized_units ? make_constants_normalized() : make_constants_SI();

    amrex::ParmParse pp(m_name);
    std::string element = "";
    amrex::Real mass_Da = 0;
    pp.query("element", element);
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

    pp.query("mass_Da", mass_Da);
    if(mass_Da != 0) {
        m_mass = phys_const.m_p * mass_Da / 1.007276466621;
    }
    pp.query("mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The plasma particle mass must be specified");

    bool ion_lev_specified = pp.query("initial_ion_level", m_init_ion_lev);
    m_can_ionize = pp.contains("ionization_product");

    pp.query("can_ionize", m_can_ionize);
    if(m_can_ionize) {
        m_neutralize_background = false; // change default
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!normalized_units,
            "Cannot use Ionization Module in normalized units");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_init_ion_lev >= 0,
            "The initial Ion level must be specified");
    }
    pp.query("neutralize_background", m_neutralize_background);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_can_ionize || !m_neutralize_background,
        "Cannot use neutralize_background for Ion plasma");

    if(!pp.query("charge", m_charge)) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_charge != 0,
            "The plasma particle charge must be specified");
    }

    if(ion_lev_specified && !m_can_ionize) {
        m_charge *= m_init_ion_lev;
    }
    pp.query("ionization_product", m_product_name);
    pp.query("density", m_density);
    pp.query("radius", m_radius);
    pp.query("hollow_core_radius", m_hollow_core_radius);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_hollow_core_radius < m_radius,
                                     "The hollow core plasma radius must not be smaller than the "
                                     "plasma radius itself");
    pp.query("parabolic_curvature", m_parabolic_curvature);
    pp.query("max_qsa_weighting_factor", m_max_qsa_weighting_factor);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(tmp_vector.size() == AMREX_SPACEDIM-1,
        "ppc is only specified in transverse directions for plasma particles, "
        "it is 1 in the longitudinal direction z. "
        "Hence, in 3D, plasma.ppc should only contain 2 values");
        for (int i=0; i<AMREX_SPACEDIM-1; i++) m_ppc[i] = tmp_vector[i];
    }
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    if (pp.query("u_mean", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_mean[idim] = loc_array[idim];
        }
    }
    if (pp.query("u_std", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_std[idim] = loc_array[idim];
        }
    }
}

void
PlasmaParticleContainer::InitData ()
{
    reserveData();
    resizeData();

    InitParticles(m_ppc, m_u_std, m_u_mean, m_density, m_radius, m_hollow_core_radius);

    m_num_exchange = TotalNumberOfParticles();
}

void
PlasmaParticleContainer::
IonizationModule (const int lev,
                  const amrex::Geometry& geom,
                  Fields& fields)
{
    HIPACE_PROFILE("PlasmaParticleContainer::IonizationModule()");

    using namespace amrex::literals;

    if (!m_can_ionize) return;
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = geom.CellSize();
    const PhysConst phys_const = make_constants_SI();

    // Loop over particle boxes with both ion and electron Particle Containers at the same time
    for (amrex::MFIter mfi_ion = MakeMFIter(lev); mfi_ion.isValid(); ++mfi_ion)
    {
        // Extract properties associated with the extent of the current box
        // Grow to capture the extent of the particle shape
        amrex::Box tilebox = mfi_ion.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});

        amrex::RealBox const grid_box{tilebox, geom.CellSize(), geom.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields
        const amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
        const amrex::MultiFab exmby(S, amrex::make_alias, Comps[WhichSlice::This]["ExmBy"], 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, Comps[WhichSlice::This]["EypBx"], 1);
        const amrex::MultiFab ez(S, amrex::make_alias, Comps[WhichSlice::This]["Ez"], 1);
        const amrex::MultiFab bx(S, amrex::make_alias, Comps[WhichSlice::This]["Bx"], 1);
        const amrex::MultiFab by(S, amrex::make_alias, Comps[WhichSlice::This]["By"], 1);
        const amrex::MultiFab bz(S, amrex::make_alias, Comps[WhichSlice::This]["Bz"], 1);
        // Extract FabArray for this box
        const amrex::FArrayBox& exmby_fab = exmby[mfi_ion];
        const amrex::FArrayBox& eypbx_fab = eypbx[mfi_ion];
        const amrex::FArrayBox& ez_fab = ez[mfi_ion];
        const amrex::FArrayBox& bx_fab = bx[mfi_ion];
        const amrex::FArrayBox& by_fab = by[mfi_ion];
        const amrex::FArrayBox& bz_fab = bz[mfi_ion];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();
        // Extract particle data
        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};
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

        const amrex::Real zmin = xyzmin[2];
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

            doGatherShapeN(xp, yp, zmin,
                           ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                           exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                           dx_arr, xyzmin_arr, lo, depos_order_xy, 0);

            const amrex::ParticleReal Exp = ExmByp + Byp * phys_const.c;
            const amrex::ParticleReal Eyp = EypBxp - Bxp * phys_const.c;
            const amrex::ParticleReal Ep = std::sqrt( Exp*Exp + Eyp*Eyp + Ezp*Ezp );

            // Compute probability of ionization p
            const amrex::Real psi_1 = ( psip[ip] *
                phys_const.q_e / (phys_const.m_e * phys_const.c * phys_const.c) ) + 1._rt;
            const amrex::Real gammap = (1.0_rt + uxp[ip] * uxp[ip] * clightsq
                                               + uyp[ip] * uyp[ip] * clightsq
                                               + psi_1 * psi_1 ) / ( 2.0_rt * psi_1 );
            const int ion_lev_loc = ion_lev[ip];
            // gamma / (psi + 1) to complete dt for QSA
            amrex::Real w_dtau = gammap / psi_1 * adk_prefactor[ion_lev_loc] *
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
        amrex::Gpu::synchronize();

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
                arrdata_elec[PlasmaIdx::psi     ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::x_prev  ][pidx] = arrdata_ion[PlasmaIdx::x_prev][ip];
                arrdata_elec[PlasmaIdx::y_prev  ][pidx] = arrdata_ion[PlasmaIdx::y_prev][ip];
                arrdata_elec[PlasmaIdx::ux_temp ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::uy_temp ][pidx] = 0._rt;
                arrdata_elec[PlasmaIdx::psi_temp][pidx] = 0._rt;
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
        amrex::Gpu::synchronize();
    }
}
