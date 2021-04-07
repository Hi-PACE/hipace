#include "Hipace.H"
#include "PlasmaParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/IonizationEnergiesTable.H"
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
    std::string element;
    pp.query("element", element);
    if (element == "electron") {
        m_charge_number = -1;
        m_mass = phys_const.m_e;
    }
    else if (element == "positron") {
        m_charge_number = 1;
        m_mass = phys_const.m_e;
    }
    else if (ion_map_ids.count(element) == 0) {
        amrex::Abort("Unknown Element\n");
    }
    pp.query("charge", m_charge_number);
    amrex::Real mass_Da;
    bool mass_in_Da = false;
    mass_in_Da = pp.query("mass_Da", mass_Da);
    if(mass_in_Da) {
        m_mass = 1.007276466621 * phys_const.m_p * mass_Da;
    }
    pp.query("mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The plasma particle mass must be specified");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_charge_number != -1024,
        "The plasma particle charge must be specified");

    pp.query("can_ionize", m_can_ionize);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_can_ionize || !normalized_units,
        "Cannot use Ionization Module in normalized units");
    if(m_can_ionize) {
        // change default value
        m_neutralize_background = false;
    }
    pp.query("neutralize_background", m_neutralize_background);

    pp.query("ionization_product", m_product_name);
    pp.query("density", m_density);
    pp.query("radius", m_radius);
    pp.query("channel_radius", m_channel_radius);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_channel_radius != 0,
        "The plasma channel radius must not be 0");
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

    InitParticles(m_ppc, m_u_std, m_u_mean, m_density, m_radius);

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
    PlasmaParticleIterator::allowMultipleMFIters(true);
    PlasmaParticleIterator pti_ion(*this, lev);
    PlasmaParticleIterator pti_elec(*m_product_pc, lev);
    while (pti_ion.isValid())
    {
        // Extract properties associated with the extent of the current box
        // Grow to capture the extent of the particle shape
        amrex::Box tilebox = pti_ion.tilebox().grow(
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
        const amrex::FArrayBox& exmby_fab = exmby[pti_ion];
        const amrex::FArrayBox& eypbx_fab = eypbx[pti_ion];
        const amrex::FArrayBox& ez_fab = ez[pti_ion];
        const amrex::FArrayBox& bx_fab = bx[pti_ion];
        const amrex::FArrayBox& by_fab = by[pti_ion];
        const amrex::FArrayBox& bz_fab = bz[pti_ion];
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
        auto& soa_ion = pti_ion.GetStructOfArrays(); // For momenta and weights
        auto& soa_elec = pti_elec.GetStructOfArrays();
        auto& aos_elec = pti_elec.GetArrayOfStructs();
        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto getPosition = GetParticlePosition<PTileType>(pti_ion.GetParticleTile());

        const amrex::Real zmin = xyzmin[2];
        const amrex::Real clightsq = 1.0_rt / ( phys_const.c * phys_const.c );

        int * const q_z = soa_ion.GetIntData(PlasmaIdx::q_z).data();
        const amrex::Real * const uxp = soa_ion.GetRealData(PlasmaIdx::ux).data();
        const amrex::Real * const uyp = soa_ion.GetRealData(PlasmaIdx::uy).data();
        const amrex::Real * const psip = soa_ion.GetRealData(PlasmaIdx::psi).data();

        // Make Ion Mask and load ADK prefactors
        // Ion Mask is necessary to only resize electron soa and aos once
        amrex::Gpu::DeviceVector<int8_t> ion_mask(pti_ion.numParticles(), 0);
        int8_t* AMREX_RESTRICT p_ion_mask = ion_mask.data();
        amrex::Gpu::DeviceScalar<long> num_new_electrons(0);
        long* AMREX_RESTRICT p_num_new_electrons = num_new_electrons.dataPtr();
        amrex::Real* AMREX_RESTRICT adk_prefactor = m_adk_prefactor.data();
        amrex::Real* AMREX_RESTRICT adk_exp_prefactor = m_adk_exp_prefactor.data();
        amrex::Real* AMREX_RESTRICT adk_power = m_adk_power.data();

        amrex::ParallelFor(pti_ion.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

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

            amrex::ParticleReal Exp = ExmByp + Byp * phys_const.c;
            amrex::ParticleReal Eyp = EypBxp - Bxp * phys_const.c;
            amrex::ParticleReal Ep = std::sqrt( Exp*Exp + Eyp*Eyp + Ezp*Ezp );

            // Compute probability of ionization p
            const amrex::Real psi_1 = ( psip[ip] *  //is m_e correct here?
                phys_const.q_e / (phys_const.m_e * phys_const.c * phys_const.c) ) + 1._rt;
            const amrex::Real gammap = (1.0_rt + uxp[ip] * uxp[ip] * clightsq
                                               + uyp[ip] * uyp[ip] * clightsq
                                               + psi_1 * psi_1 ) / ( 2.0_rt * psi_1 );
            const int ion_lev = q_z[ip];
            // gamma / (psi + 1) to complete dt for QSA
            amrex::Real w_dtau = gammap / psi_1 * adk_prefactor[ion_lev] *
                std::pow(Ep, adk_power[ion_lev]) *
                std::exp( adk_exp_prefactor[ion_lev]/Ep );
            amrex::Real p = 1._rt - std::exp( - w_dtau );

            amrex::Real random_draw = amrex::Random();
            if (random_draw < p)
            {
                q_z[ip] += 1;
                p_ion_mask[ip] = 1;
                ++(*p_num_new_electrons);
            }
        });
        amrex::Gpu::synchronize();

        if(Hipace::m_verbose >= 3) {
            amrex::Print() << "Number of ionized Plasma Particles: "
            << num_new_electrons.dataValue() << "\n";
        }

        auto old_size = soa_elec.size();
        auto new_size = old_size + num_new_electrons.dataValue();
        soa_elec.resize(new_size);
        aos_elec.resize(new_size);

        // Load electron soa and aos after resize
        ParticleType* pstruct_elec = aos_elec().data();
        int procID = amrex::ParallelDescriptor::MyProc();
        amrex::Gpu::DeviceScalar<long> ip_elec(0);
        long* AMREX_RESTRICT p_ip_elec = ip_elec.dataPtr();
        long pid_start = ParticleType::NextID();
        ParticleType::NextID(pid_start + num_new_electrons.dataValue());

        auto arrdata_ion = soa_ion.realarray();
        auto arrdata_elec = soa_elec.realarray();
        auto int_arrdata_elec = soa_elec.intarray();

        int charge_number = m_product_pc->m_charge_number;

        amrex::ParallelFor(pti_ion.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

            if(p_ion_mask[ip]) {
                long pid = (* p_ip_elec)++;
                long pidx = pid + old_size;

                // Copy ion data to new electron
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                pstruct_elec[pidx].id()   = pid_start + pid;
                pstruct_elec[pidx].cpu()  = procID;
                pstruct_elec[pidx].pos(0) = xp;
                pstruct_elec[pidx].pos(1) = yp;
                pstruct_elec[pidx].pos(2) = zp;

                arrdata_elec[PlasmaIdx::w        ][pidx] = arrdata_ion[PlasmaIdx::w        ][ip];
                arrdata_elec[PlasmaIdx::w0       ][pidx] = arrdata_ion[PlasmaIdx::w0       ][ip];
                arrdata_elec[PlasmaIdx::ux       ][pidx] = arrdata_ion[PlasmaIdx::ux       ][ip];
                arrdata_elec[PlasmaIdx::uy       ][pidx] = arrdata_ion[PlasmaIdx::uy       ][ip];
                arrdata_elec[PlasmaIdx::psi      ][pidx] = arrdata_ion[PlasmaIdx::psi      ][ip];
                arrdata_elec[PlasmaIdx::x_prev   ][pidx] = arrdata_ion[PlasmaIdx::x_prev   ][ip];
                arrdata_elec[PlasmaIdx::y_prev   ][pidx] = arrdata_ion[PlasmaIdx::y_prev   ][ip];
                arrdata_elec[PlasmaIdx::ux_temp  ][pidx] = arrdata_ion[PlasmaIdx::ux_temp  ][ip];
                arrdata_elec[PlasmaIdx::uy_temp  ][pidx] = arrdata_ion[PlasmaIdx::uy_temp  ][ip];
                arrdata_elec[PlasmaIdx::psi_temp ][pidx] = arrdata_ion[PlasmaIdx::psi_temp ][ip];
                arrdata_elec[PlasmaIdx::x0       ][pidx] = arrdata_ion[PlasmaIdx::x0       ][ip];
                arrdata_elec[PlasmaIdx::y0       ][pidx] = arrdata_ion[PlasmaIdx::y0       ][ip];
                int_arrdata_elec[PlasmaIdx::q_z  ][pidx] = charge_number;
            }
        });
        amrex::Gpu::synchronize();

        ++pti_ion;
        ++pti_elec;
    }
    PlasmaParticleIterator::allowMultipleMFIters(false);
}
