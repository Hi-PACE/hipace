#include "PlasmaParticleContainer.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

PlasmaParticleContainer::PlasmaParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainer<0,0,PlasmaIdx::nattribs>(amr_core->GetParGDB())
{
    amrex::ParmParse pp("plasma");
    pp.query("density", m_density);
    pp.query("radius", m_radius);
    pp.query("max_qsa_weighting_factor", m_max_qsa_weighting_factor);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(tmp_vector.size() == AMREX_SPACEDIM-1,
                                         "ppc is only specified in transverse directions for plasma particles, it is 1 in the longitudinal direction z. Hence, in 3D, plasma.ppc should only contain 2 values");
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
    pp.query("sort_int", m_sort_int);
    if (pp.query("sort_bin_size", m_sort_bin_size))
    {
        m_sort_bin = {m_sort_bin_size, m_sort_bin_size, 1};
    }
}

void
PlasmaParticleContainer::InitData ()
{
    reserveData();
    resizeData();

    InitParticles(m_ppc,m_u_std, m_u_mean, m_density, m_radius);

    m_num_exchange = TotalNumberOfParticles();
}

void
PlasmaParticleContainer::RedistributeSlice (int const lev)
{
    HIPACE_PROFILE("PlasmaParticleContainer::RedistributeSlice()");

    using namespace amrex::literals;
    const auto plo    = Geom(lev).ProbLoArray();
    const auto phi    = Geom(lev).ProbHiArray();
    const auto is_per = Geom(lev).isPeriodicArray();
    AMREX_ALWAYS_ASSERT(is_per[0] == is_per[1]);

    amrex::GpuArray<int,AMREX_SPACEDIM> const periodicity = {true, true, false};
    // Loop over particle boxes
    for (PlasmaParticleIterator pti(*this, lev); pti.isValid(); ++pti)
    {

        // Extract particle properties
        auto& aos = pti.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin();
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        amrex::Real * const wp = soa.GetRealData(PlasmaIdx::w).data();

        // Loop over particles and handle particles outside of the box
        amrex::ParallelFor(
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

                const bool shifted = enforcePeriodic(pos_structs[ip], plo, phi, periodicity);

                if (shifted && !is_per[0]) wp[ip] = 0.0_rt;

            }
            );
    }
}

void
PlasmaParticleContainer::BinSort (int step)
{
    if (step % m_sort_int != 0) return;
    SortParticlesByBin(m_sort_bin);
}
