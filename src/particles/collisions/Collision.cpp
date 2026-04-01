#include "Collision.H"
#include "Hipace.H"
#include "particles/sorting/TileSort.H"
#include "ShuffleFisherYates.H"
#include "utils/Constants.H"
#include "utils/IonizationEnergiesTable.H"
#include "utils/GPUUtil.H"
#include "ImpactIonization.H"
#include "ImpactIonizationSigma.H"

using namespace amrex::literals;

// Binding energies - reduced to H, N, He, Ar (eV)
// Innermost shell first, progressing outward
namespace {
    constexpr amrex::Real bindingEnergies[28] = {
        /* Z=1  (H)  — 1 electron  */   13.5981,
        /* Z=7  (N)  — 7 electrons */   403.78, 403.78,
                                        20.33, 20.33,
                                        14.534, 14.534, 14.534,
        /* Z=2  (He) — 2 electrons */   24.588, 24.588,
        /* Z=18 (Ar) — 18 electrons */  3206.2, 3206.2,
                                        324.2, 324.2,
                                        247.74, 247.74, 247.74, 247.74, 247.74, 247.74,
                                        29.24, 29.24,
                                        15.76, 15.76, 15.76, 15.76, 15.76, 15.76
    };  // Very structure specific to ImpactIonizationSigma.H

    constexpr amrex::Real ionizationEnergies[28] = {
        /* Z=1  (H)  */ 13.598434005136,

        /* Z=7  (N)  : 1s, 1s, 2s, 2s, 2p, 2p, 2p */
                        667.04609, 552.06731,
                        97.89013,  77.4735,
                        47.4453,   29.60125, 14.53413,

        /* Z=2  (He) : 1s, 1s */
                        54.41776311, 24.587387936,

        /* Z=18 (Ar) : 1s,1s,2s,2s,2p x6,3s x2,3p x6 */
                        4426.2227, 4120.6655,
                        918.374,   855.47,
                        755.13,    685.47,   619.0,
                        540.4,     479.76,   422.60,
                        143.457,   124.41,
                        91.290,    74.84,    59.58,
                        40.735,    27.62967, 15.7596112
    };
}

void
Collision::ReadParameters(
    const std::vector<std::string>& plasma_species_names,
    std::string const collision_name)
{
    amrex::ParmParse pp(collision_name);

    getWithParser(pp, "type", m_collision_type);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_collision_type == "electron_impact" ||
        m_collision_type == "ion_impact",
        "Unknown collision type"
    );
    // The projectile can be either an electron or an ion
    // Target particle is always a neutral/ion
    // The user is asked to always input the target type as the second specie
    m_has_collision_product = m_collision_type == "electron_impact" || m_collision_type == "ion_impact";

    getWithParser(pp, "species1", m_inout_species1_name);
    getWithParser(pp, "species2", m_inout_species2_name);

    if (m_has_collision_product) {
        getWithParser(pp, "species3", m_out_species3_name);
    }

    if (m_collision_type == "electron_impact") {
        // Initialize binding energies for collision calculation
        m_binding_energies.resize(28);
        m_ionization_energies.resize(28);
        for (int i=0; i<28 ; ++i) {
            m_binding_energies[i] = bindingEnergies[i];
            m_ionization_energies[i] = ionizationEnergies[i];
        }
        m_binding_energies.copyToDeviceAsync();
        m_ionization_energies.copyToDeviceAsync();
    }
    else {

    }
    // Plasma physical element name
    amrex::ParmParse pp_s2(m_inout_species2_name);
    getWithParser(pp_s2, "element", m_physical_element);
}

void
Collision::doCollision (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma)
{
    if (m_collision_type == "electron_impact") {
        doCollisionA(lev, geom, multi_plasma);
    } else {
        doCollisionB(lev, geom, multi_plasma);
    }
}
// SI units are assumed (!!!)
// Different species colliding (electron impact)
void
Collision::doCollisionA (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma)
{
    constexpr amrex::Real c2 = PhysConstSI::c * PhysConstSI::c;
    constexpr amrex::Real inv_c = 1.0_rt / PhysConstSI::c;
    constexpr amrex::Real inv_c2 = 1.0_rt / (PhysConstSI::c * PhysConstSI::c);

    // auto p_crosssection_data = m_crosssection_data.data();   !!!
    const amrex::Real* p_binding_energies = m_binding_energies.data();
    const amrex::Real* p_ionization_energies = m_ionization_energies.data();

    auto& species1 = multi_plasma.GetPlasma(m_inout_species1_name);
    auto& species2 = multi_plasma.GetPlasma(m_inout_species2_name);
    auto& species3 = multi_plasma.GetPlasma(m_out_species3_name);

    auto m1 = species1.GetMass();
    auto m2 = species2.GetMass();
    auto m3 = species3.GetMass();

    const int ion_element_id = ion_map_ids[m_physical_element];
    const int ion_atomic_number = ion_atomic_numbers[ion_element_id];

    const amrex::Real inv_dV = geom.InvCellSize(0)*geom.InvCellSize(1)*geom.InvCellSize(2);
    const amrex::Real dt = geom.CellSize(2) * inv_c;

    doCollisionImp(lev, geom, multi_plasma,
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2,
                              int N1, int N2, int icoll,
                              amrex::RandomEngine const& engine)
        {
            // Collision function
            // clean from unused quantities
            amrex::Real ux1 = ptd1.rdata(PlasmaIdx::ux_half_step)[i1];
            amrex::Real uy1 = ptd1.rdata(PlasmaIdx::uy_half_step)[i1];
            amrex::Real psi1 = ptd1.rdata(PlasmaIdx::psi_half_step)[i1];
            const amrex::Real w1 = ptd1.rdata(PlasmaIdx::w)[i1];

            amrex::Real ux2 = ptd2.rdata(PlasmaIdx::ux_half_step)[i2];
            amrex::Real uy2 = ptd2.rdata(PlasmaIdx::uy_half_step)[i2];
            amrex::Real psi2 = ptd2.rdata(PlasmaIdx::psi_half_step)[i2];
            const amrex::Real w2 = ptd2.rdata(PlasmaIdx::w)[i2];
            int ion_lev2 = ptd2.idata(PlasmaIdx::ion_lev)[i2];

            if (ion_lev2 > 0) {
                // Already ionized, do not collide
                return false;
            }

            // particle's Lorentz factor
            amrex::Real g1 = plasma_gamma(ux1, uy1, psi1, 1._rt / psi1, /* Assumes Aabssq == 0 */ 0._rt);
            amrex::Real g2 = plasma_gamma(ux2, uy2, psi2, 1._rt / psi2, /* Assumes Aabssq == 0 */ 0._rt);

            // Convert from pseudo-potential to momentum
            amrex::Real uz1 = plasma_uz(g1, psi1);
            amrex::Real uz2 = plasma_uz(g2, psi2);

            // In the longitudinal push of plasma particles, the dt is different for each particle
            // The dt applied for collision probability is the average (in the lab frame) of these dts
            const amrex::Real dt_fac = 0.5_rt * (g1/psi1 + g2/psi2);

            // Rescaling of the particle weights according to eq (22)-(23),
            // Higginson et al., Journal of Computational Physics 413 (2020)
            int N12 = amrex::max(N1,N2);
            int D1;
            int D2;

            if (N1>=N2) {
                D1 = 1;
            } else {
                D1 = (N2/N1) + (icoll < (N2 % N1) ? 1 : 0);
            }

            if (N2>=N1) {
                D2 = 1;
            } else {
                D2 = (N1/N2) + (icoll < (N1 % N2) ? 1 : 0);
            }

            amrex::Real w1r = w1 / amrex::max(D1,D2);
            amrex::Real w2r = w2 / amrex::max(D1,D2);

            const amrex::Real diffx = amrex::Math::abs(ux1-ux2);
            const amrex::Real diffy = amrex::Math::abs(uy1-uy2);
            const amrex::Real diffz = amrex::Math::abs(uz1-uz2);
            const amrex::Real diffm = std::sqrt(diffx*diffx+diffy*diffy+diffz*diffz);
            const amrex::Real summ = std::sqrt(ux1*ux1+uy1*uy1+uz1*uz1) + std::sqrt(ux2*ux2+uy2*uy2+uz2*uz2);
            // If g = u1 - u2 = 0, do not collide.
            // Or if the relative difference is less than 1.0e-10.
            if ( diffm < std::numeric_limits<amrex::Real>::min() || diffm/summ < 1.0e-10 ) {
                return false;
            }
            // Compute the particles relative velocity (lab frame)
            const amrex::Real u1u2 = ux1*ux2+uy1*uy2+uz1*uz2;
            const amrex::Real grel = g1*g2-u1u2;
            const amrex::Real inv_grel2 = 1/(grel*grel);
            const amrex::Real vrel = PhysConstSI::c * std::sqrt(1-inv_grel2);

            // Fetch the cross section
            const amrex::Real Krel = (grel - 1.0) * m1 * c2 / PhysConstSI::q_e;    // (eV)
            auto cs_data = coll_ion::sigma_E(ion_atomic_number, ion_lev2, Krel, p_binding_energies, p_ionization_energies);
            auto sigma = cs_data.sigma;

            // The ionization mechanism is based on the algorithm from Perez et al., Phys.Plasmas.19.083104 (2012)
            // The weights are rescaled according to eq (22)-(23),
            // Higginson et al., Journal of Computational Physics 413 (2020)

            // Ionization probability
            const auto Pion = 1 - std::exp(-vrel * N12 * amrex::max(w1r,w2r) * inv_dV * sigma * dt*dt_fac);
            // Get random numbers
            auto r = amrex::Random(engine);
            if (Pion > r) {
                // Rejection method according to the adaptation of eq (14) in the ionization model, 
                // from Perez et al., Phys.Plasmas.19.083104 (2012)
                r = amrex::Random(engine);
                return ( w1r > r*amrex::max(w1r,w2r) );
            } else {
                return false;
            }
        },
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2, auto ptd3, int i3,
                              int N1, int N2, int icoll,
                              amrex::RandomEngine const& engine)
        {
            // Ionization function
            // Ionization occurs if the condition on Pion is satisfied
            // A new electron is created following the rejection method
            amrex::Real ux1 = ptd1.rdata(PlasmaIdx::ux_half_step)[i1];
            amrex::Real uy1 = ptd1.rdata(PlasmaIdx::uy_half_step)[i1];
            amrex::Real psi1 = ptd1.rdata(PlasmaIdx::psi_half_step)[i1];
            const amrex::Real w1 = ptd1.rdata(PlasmaIdx::w)[i1];
            const int ion_lev1 = ptd1.idata(PlasmaIdx::ion_lev)[i1];

            amrex::Real ux2 = ptd2.rdata(PlasmaIdx::ux_half_step)[i2];
            amrex::Real uy2 = ptd2.rdata(PlasmaIdx::uy_half_step)[i2];
            amrex::Real psi2 = ptd2.rdata(PlasmaIdx::psi_half_step)[i2];
            const amrex::Real w2 = ptd2.rdata(PlasmaIdx::w)[i2];
            int ion_lev2 = ptd2.idata(PlasmaIdx::ion_lev)[i2];

            // Initialize new electron properties
            ptd3.id(i3) = 4;
            ptd3.cpu(i3) = 0;
            ptd3.rdata(PlasmaIdx::x)[i3] = ptd2.rdata(PlasmaIdx::x)[i2];
            ptd3.rdata(PlasmaIdx::y)[i3] = ptd2.rdata(PlasmaIdx::y)[i2];
            ptd3.rdata(PlasmaIdx::x_prev)[i3] = ptd2.rdata(PlasmaIdx::x_prev)[i2];
            ptd3.rdata(PlasmaIdx::y_prev)[i3] = ptd2.rdata(PlasmaIdx::y_prev)[i2];
            ptd3.rdata(PlasmaIdx::w)[i3] = ptd2.rdata(PlasmaIdx::w)[i2];
            ptd3.idata(PlasmaIdx::ion_lev)[i3] = 0;
            // Quantities to be updated after collision
            amrex::Real ux3 = 0._rt;
            amrex::Real uy3 = 0._rt;
            amrex::Real psi3 = 1._rt;
            amrex::Real uz3 = 0._rt;
            ptd3.rdata(PlasmaIdx::ux)[i3] = ux3;
            ptd3.rdata(PlasmaIdx::uy)[i3] = uy3;
            ptd3.rdata(PlasmaIdx::psi)[i3] = psi3;
            ptd3.rdata(PlasmaIdx::ux_half_step)[i3] = ux3;
            ptd3.rdata(PlasmaIdx::uy_half_step)[i3] = uy3;
            ptd3.rdata(PlasmaIdx::psi_half_step)[i3] = psi3;

            // particle's Lorentz factor
            amrex::Real g1 = plasma_gamma(ux1, uy1, psi1, 1._rt / psi1, /* Assumes Aabssq == 0 */ 0._rt);
            amrex::Real g2 = plasma_gamma(ux2, uy2, psi2, 1._rt / psi2, /* Assumes Aabssq == 0 */ 0._rt);

            // Convert from pseudo-potential to momentum
            amrex::Real uz1 = plasma_uz(g1, psi1);
            amrex::Real uz2 = plasma_uz(g2, psi2);

            // Fetch the ionization energy
            const amrex::Real u1u2 = ux1*ux2+uy1*uy2+uz1*uz2;
            const amrex::Real grel = g1*g2-u1u2;
            const amrex::Real Krel = (grel - 1.0) * m1 * c2 / PhysConstSI::q_e;    // (eV)
            auto cs_data = coll_ion::sigma_E(ion_atomic_number, ion_lev2, Krel, p_binding_energies, p_ionization_energies);
            auto Eion_eV = cs_data.ionization_en;

            // Rescaling of the particle weights according to eq (22)-(23),
            // Higginson et al., Journal of Computational Physics 413 (2020)
            int N12 = amrex::max(N1,N2);
            int D1;
            int D2;

            if (N1>=N2) {
                D1 = 1;
            } else {
                D1 = (N2/N1) + (icoll < (N2 % N1) ? 1 : 0);
            }

            if (N2>=N1) {
                D2 = 1;
            } else {
                D2 = (N1/N2) + (icoll < (N1 % N2) ? 1 : 0);
            }

            amrex::Real w1r = w1 / amrex::max(D1,D2);
            amrex::Real w2r = w2 / amrex::max(D1,D2);

            ImpactIonization(
                ux1, uy1, uz1, g1,
                ux2, uy2, uz2, g2, ion_lev2,
                ux3, uy3, uz3,
                m1, w1r, m2, w2r, m3,
                Eion_eV, c2, inv_c, inv_c2,
                engine
            );

            // Update particle properties
            ptd1.rdata(PlasmaIdx::ux_half_step)[i1] = ux1;
            ptd1.rdata(PlasmaIdx::uy_half_step)[i1] = uy1;
            ptd1.rdata(PlasmaIdx::psi_half_step)[i1] = plasma_psi(ux1, uy1, uz1, /* Assumes Aabssq == 0 */ 0._rt);

            ptd2.rdata(PlasmaIdx::ux_half_step)[i2] = ux2;
            ptd2.rdata(PlasmaIdx::uy_half_step)[i2] = uy2;
            ptd2.rdata(PlasmaIdx::psi_half_step)[i2] = plasma_psi(ux2, uy2, uz2, /* Assumes Aabssq == 0 */ 0._rt);
            ptd2.idata(PlasmaIdx::ion_lev)[i2] = ion_lev2;

            ptd3.rdata(PlasmaIdx::ux)[i3] = ux3;
            ptd3.rdata(PlasmaIdx::uy)[i3] = uy3;
            ptd3.rdata(PlasmaIdx::psi)[i3] = plasma_psi(ux3, uy3, uz3, /* Assumes Aabssq == 0 */ 0._rt);
            ptd3.rdata(PlasmaIdx::ux_half_step)[i3] = ux3;
            ptd3.rdata(PlasmaIdx::uy_half_step)[i3] = uy3;
            ptd3.rdata(PlasmaIdx::psi_half_step)[i3] = plasma_psi(ux3, uy3, uz3, /* Assumes Aabssq == 0 */ 0._rt);
        });
}

// Same species colliding (Ion impact)
void
Collision::doCollisionB (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma)
{
    doCollisionImp(lev, geom, multi_plasma,
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2,
                              int N1, int N2, int icoll,
                              amrex::RandomEngine const& engine)
        {
            return false;
        },
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2, auto ptd3, int i3,
                              int N1, int N2, int icoll,
                              amrex::RandomEngine const& engine)
        {

        });
}

template <class F, class G>
void
Collision::doCollisionImp (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        F const& collision_function,
        G const& ionization_function)
{
    // assume the two species are different for now
    auto& species1 = multi_plasma.GetPlasma(m_inout_species1_name);
    auto& species2 = multi_plasma.GetPlasma(m_inout_species2_name);

    for (PlasmaParticleIterator pti(species1); pti.isValid(); ++pti) {
        amrex::removeInvalidParticles(species1.ParticlesAt(0, pti));
        amrex::removeInvalidParticles(species2.ParticlesAt(0, pti));
    }

    PlasmaBins bins1 = findParticlesInEachTile(geom.Domain(), 1, species1, geom);
    PlasmaBins bins2 = findParticlesInEachTile(geom.Domain(), 1, species2, geom);

    // offset: start/end positions of particles for each cell
    // perm: permutation array mapping cell entries to particle indices
    auto offset1 = bins1.offsetsPtr();
    auto offset2 = bins2.offsetsPtr();
    auto perm1 = bins1.permutationPtr();
    auto perm2 = bins2.permutationPtr();

    const int num_cells = bins1.numBins();

    amrex::Gpu::DeviceVector<int> num_ind_pairs(num_cells, 0);
    auto p_num_ind_pairs = num_ind_pairs.dataPtr();

    const int total_ind_pairs = amrex::Scan::PrefixSum<int>(num_cells,
        [=] AMREX_GPU_DEVICE (int i) {
            int N1 = offset1[i+1] - offset1[i];
            int N2 = offset2[i+1] - offset2[i];
            return std::min(N1, N2);
        },
        [=] AMREX_GPU_DEVICE (int i, int s) {
            p_num_ind_pairs[i] = s;
        },
        amrex::Scan::Type::exclusive, amrex::Scan::retSum
    );

    amrex::ParallelForRNG(num_cells,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) {
            ShuffleFisherYates(perm1, offset1[i], offset1[i+1], engine);
            ShuffleFisherYates(perm2, offset2[i], offset2[i+1], engine);
        }
    );

    // loop over tiles
    for (PlasmaParticleIterator pti(species1); pti.isValid(); ++pti) {

        auto& ptile1 = species1.ParticlesAt(0, pti);
        auto& ptile2 = species2.ParticlesAt(0, pti);

        auto ptd1 = ptile1.getParticleTileData();
        auto ptd2 = ptile2.getParticleTileData();

        const int np1 = ptile1.numParticles();
        const int np2 = ptile2.numParticles();

        amrex::Gpu::DeviceVector<int> flag1;
        amrex::Gpu::DeviceVector<int> flag2;

        if (m_has_collision_product) {
            flag1.resize(np1, 0);
            flag2.resize(np2, 0);
        }

        auto p_flag1 = flag1.dataPtr();
        auto p_flag2 = flag2.dataPtr();

        amrex::Gpu::DeviceVector<amrex::Real> cell_weight1(num_cells);
        amrex::Gpu::DeviceVector<amrex::Real> cell_weight2(num_cells);
        auto p_cell_weight1 = cell_weight1.dataPtr();
        auto p_cell_weight2 = cell_weight2.dataPtr();

        amrex::ParallelFor(2*num_cells,
            [=] AMREX_GPU_DEVICE (int icell) {
                if (icell < num_cells) {
                    auto start = offset1[icell];
                    auto stop = offset1[icell+1];
                    amrex::Real loc_cell_weight1 = 0;
                    for (int idx1 = start; idx1 < stop; ++idx1) {
                        loc_cell_weight1 += ptd1.rdata(PlasmaIdx::w)[perm1[idx1]];
                    }
                    p_cell_weight1[icell] = loc_cell_weight1;
                } else {
                    icell -= num_cells;
                    auto start = offset2[icell];
                    auto stop = offset2[icell+1];
                    amrex::Real loc_cell_weight2 = 0;
                    for (int idx2 = start; idx2 < stop; ++idx2) {
                        loc_cell_weight2 += ptd2.rdata(PlasmaIdx::w)[perm2[idx2]];
                    }
                    p_cell_weight2[icell] = loc_cell_weight2;
                }
            }
        );

        // loop over independent pairs
        amrex::ParallelForRNG(total_ind_pairs,
            [=] AMREX_GPU_DEVICE (int ipair, amrex::RandomEngine const& engine){
                const int icell = amrex::bisect(p_num_ind_pairs, 0, num_cells, ipair);

                const int offset1_start = offset1[icell];
                const int offset1_stop = offset1[icell+1];
                const int offset2_start = offset2[icell];
                const int offset2_stop = offset2[icell+1];

                const int icoll = ipair - p_num_ind_pairs[icell];

                const int N1 = offset1_stop - offset1_start;
                const int N2 = offset2_stop - offset2_start;

                //const amrex::Real loc_cell_weight1 = p_cell_weight1[icell];
                //const amrex::Real loc_cell_weight2 = p_cell_weight2[icell];

                int idx1 = icoll;
                int idx2 = icoll;
                while (idx1 < N1 && idx2 < N2) {
                    const int j1 = perm1[offset1_start + idx1];
                    const int j2 = perm2[offset2_start + idx2];

                    const bool make_new_particle = collision_function(
                        ptd1, j1, ptd2, j2,
                        N1, N2, icoll,
                        engine
                    );

                    if (N2 < N1) {
                        idx1 += N2;
                        if (make_new_particle) {
                            p_flag1[j1] = 1;
                        }
                    } else {
                        idx2 += N1;
                        if (make_new_particle) {
                            p_flag2[j2] = 1;
                        }
                    }
                }
            }
        );

        if (!m_has_collision_product) {
            continue;
        }

        const int num_new_particles = amrex::Scan::PrefixSum<int>(np1+np2,
            [=] AMREX_GPU_DEVICE (int i) {
                if (i < np1) {
                    return p_flag1[i];
                } else {
                    i -= np1;
                    return p_flag2[i];
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int s) {
                if (i < np1) {
                    p_flag1[i] = p_flag1[i] == 0 ? -1 : s;
                } else {
                    i -= np1;
                    p_flag2[i] = p_flag2[i] == 0 ? -1 : s;
                }
            },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        auto& species3 = multi_plasma.GetPlasma(m_out_species3_name);
        auto& ptile3 = species3.ParticlesAt(0, pti);
        int old_size3 = ptile3.size();
        ptile3.resize(old_size3 + num_new_particles);

        // get new ptd after resize in case species3 is the same as species1 or 2
        ptd1 = ptile1.getParticleTileData();
        ptd2 = ptile2.getParticleTileData();
        auto ptd3 = ptile3.getParticleTileData();

        amrex::ParallelForRNG(total_ind_pairs,
            [=] AMREX_GPU_DEVICE (int ipair, amrex::RandomEngine const& engine){
                const int icell = amrex::bisect(p_num_ind_pairs, 0, num_cells, ipair);

                const int offset1_start = offset1[icell];
                const int offset1_stop = offset1[icell+1];
                const int offset2_start = offset2[icell];
                const int offset2_stop = offset2[icell+1];

                const int icoll = ipair - p_num_ind_pairs[icell];

                const int N1 = offset1_stop - offset1_start;
                const int N2 = offset2_stop - offset2_start;

                int idx1 = icoll;
                int idx2 = icoll;
                while (idx1 < N1 && idx2 < N2) {
                    const int j1 = perm1[offset1_start + idx1];
                    const int j2 = perm2[offset2_start + idx2];
                    const int new_part_idx = N2 < N1 ? p_flag1[j1] : p_flag2[j2];

                    if (new_part_idx >= 0) {
                        ionization_function(
                            ptd1, j1, ptd2, j2, ptd3, old_size3 + new_part_idx,
                            N1, N2, icoll,
                            engine
                        );
                    }

                    if (N2 < N1) {
                        idx1 += N2;
                    } else {
                        idx2 += N1;
                    }
                }
            }
        );

        amrex::Gpu::streamSynchronize();
    }
}
