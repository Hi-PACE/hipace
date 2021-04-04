#include "Hipace.H"
#include "PlasmaParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"
#include "pusher/PlasmaParticleAdvance.H"
#include "pusher/BeamParticleAdvance.H"
#include "pusher/FieldGather.H"
#include "pusher/GetAndSetPosition.H"
#include <cmath>

namespace
{
    bool QueryElementSetChargeMass (amrex::ParmParse& pp, int& charge, amrex::Real& mass)
    {
        // normalized_units is directly queried here so we can defined the appropriate PhysConst
        // locally. We cannot use Hipace::m_phys_const as it has not been initialized when the
        // PlasmaParticleContainer constructor is called.
        amrex::ParmParse pph("hipace");
        bool normalized_units = false;
        pph.query("normalized_units", normalized_units);
        PhysConst phys_const = normalized_units ? make_constants_normalized() : make_constants_SI();

        std::string element;
        bool element_is_specified = pp.query("element", element);
        if (element_is_specified){
            if (element == "electron"){
                charge = -1;
                mass = phys_const.m_e;
            } else if (element == "proton"){
                charge = 1;
                mass = phys_const.m_p;
            } else {
                amrex::Abort("unknown plasma species. Options are: electron and H.");
            }

    }
        return element_is_specified;
    }
}

void
PlasmaParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    pp.query("charge", m_charge_number);
    pp.query("mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        QueryElementSetChargeMass(pp, m_charge_number, m_mass) ^
        (pp.query("charge", m_charge_number) && pp.query("mass", m_mass)),
        "Plasma: must specify EITHER <species>.element OR <species>.charge and <species>.mass");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The plasma particle mass must not be 0");

    pp.query("neutralize_background", m_neutralize_background);
    pp.query("can_ionize", m_can_ionize);
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

    // Loop over particle boxes
    PlasmaParticleIterator::allowMultipleMFIters(true);
    PlasmaParticleIterator pti_p(*m_product_pc, lev);
    for (PlasmaParticleIterator pti(*this, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        // Grow to capture the extent of the particle shape
        amrex::Box tilebox = pti.tilebox().grow(
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
        const amrex::FArrayBox& exmby_fab = exmby[pti];
        const amrex::FArrayBox& eypbx_fab = eypbx[pti];
        const amrex::FArrayBox& ez_fab = ez[pti];
        const amrex::FArrayBox& bx_fab = bx[pti];
        const amrex::FArrayBox& by_fab = by[pti];
        const amrex::FArrayBox& bz_fab = bz[pti];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};
        const int depos_order_xy = Hipace::m_depos_order_xy;
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        auto& soa_p = pti_p.GetStructOfArrays();
        auto& aos_p = pti_p.GetArrayOfStructs();
        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto getPosition = GetParticlePosition<PTileType>(pti.GetParticleTile());

        const amrex::Real zmin = xyzmin[2];
        const amrex::Real clightsq = 1.0_rt / ( phys_const.c * phys_const.c );

        int * const q_z = soa.GetIntData(PlasmaIdx::q_z).data();
        const amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        const amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        const amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();

        amrex::Gpu::DeviceVector<int8_t> IonMask(pti.numParticles(), 0);
        int8_t* AMREX_RESTRICT p_IonMask = IonMask.data();
        amrex::Gpu::DeviceScalar<long> num_new_electrons(0);
        long* AMREX_RESTRICT p_num_new_electrons = num_new_electrons.dataPtr();
        amrex::Real* AMREX_RESTRICT adk_prefactor = m_adk_prefactor.data();
        amrex::Real* AMREX_RESTRICT adk_exp_prefactor = m_adk_exp_prefactor.data();
        amrex::Real* AMREX_RESTRICT adk_power = m_adk_power.data();

        amrex::ParallelFor(pti.numParticles(),
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
            amrex::Real w_dtau = gammap / psi_1 * adk_prefactor[ion_lev] *
                std::pow(Ep, adk_power[ion_lev]) *
                std::exp( adk_exp_prefactor[ion_lev]/Ep );
            amrex::Real p = 1._rt - std::exp( - w_dtau );

            amrex::Real random_draw = amrex::Random();
            if (random_draw < p)
            {
                q_z[ip] += 1;
                p_IonMask[ip] = 1;
                ++(*p_num_new_electrons);
            }
        });
        amrex::Gpu::synchronize();

        if(Hipace::m_verbose >= 3) {
            amrex::Print() << "Number of ionized Plasma Particles: "
            << num_new_electrons.dataValue() <<"\n";
        }

        auto old_size = soa.size();
        auto new_size = old_size + num_new_electrons.dataValue();
        soa_p.resize(new_size);
        aos_p.resize(new_size);

        ParticleType* pstruct = aos_p().data();
        int procID = amrex::ParallelDescriptor::MyProc();
        amrex::Gpu::DeviceScalar<long> ip_p(0);
        long* AMREX_RESTRICT p_ip_p = ip_p.dataPtr();
        long pid_start = ParticleType::NextID();
        ParticleType::NextID(pid_start + num_new_electrons.dataValue());

        auto arrdata = soa.realarray();
        auto arrdata_p = soa_p.realarray();
        auto int_arrdata_p = soa_p.intarray();

        int charge_number = m_product_pc->m_charge_number;

        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

            if(p_IonMask[ip]) {
                long pid = (* p_ip_p)++;
                long pidx = pid + old_size;

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                pstruct[pidx].id()   = pid_start + pid;
                pstruct[pidx].cpu()  = procID;
                pstruct[pidx].pos(0) = xp;
                pstruct[pidx].pos(1) = yp;
                pstruct[pidx].pos(2) = zp;

                arrdata_p[PlasmaIdx::w        ][pidx] = arrdata[PlasmaIdx::w        ][ip];
                arrdata_p[PlasmaIdx::w0       ][pidx] = arrdata[PlasmaIdx::w0       ][ip];
                arrdata_p[PlasmaIdx::ux       ][pidx] = arrdata[PlasmaIdx::ux       ][ip];
                arrdata_p[PlasmaIdx::uy       ][pidx] = arrdata[PlasmaIdx::uy       ][ip];
                arrdata_p[PlasmaIdx::psi      ][pidx] = arrdata[PlasmaIdx::psi      ][ip];
                arrdata_p[PlasmaIdx::x_prev   ][pidx] = arrdata[PlasmaIdx::x_prev   ][ip];
                arrdata_p[PlasmaIdx::y_prev   ][pidx] = arrdata[PlasmaIdx::y_prev   ][ip];
                arrdata_p[PlasmaIdx::ux_temp  ][pidx] = arrdata[PlasmaIdx::ux_temp  ][ip];
                arrdata_p[PlasmaIdx::uy_temp  ][pidx] = arrdata[PlasmaIdx::uy_temp  ][ip];
                arrdata_p[PlasmaIdx::psi_temp ][pidx] = arrdata[PlasmaIdx::psi_temp ][ip];
                arrdata_p[PlasmaIdx::x0       ][pidx] = arrdata[PlasmaIdx::x0       ][ip];
                arrdata_p[PlasmaIdx::y0       ][pidx] = arrdata[PlasmaIdx::y0       ][ip];
                int_arrdata_p[PlasmaIdx::q_z  ][pidx] = charge_number;
            }
        });
        amrex::Gpu::synchronize();

        ++pti_p;
    }
}
