#include "Collision.H"
#include "Hipace.H"
#include "particles/sorting/TileSort.H"
#include "ShuffleFisherYates.H"

void
Collision::ReadParameters(
    const std::vector<std::string>& plasma_species_names,
    std::string const collision_name)
{
    amrex::ParmParse pp(collision_name);

    getWithParser(pp, "type", m_collision_type);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_collision_type == "a" ||
        m_collision_type == "b",
        "Unknown collision type"
    );

    m_has_collision_product = m_collision_type == "a";

    getWithParser(pp, "species1", m_inout_species1_name);
    getWithParser(pp, "species2", m_inout_species2_name);

    if (m_has_collision_product) {
        getWithParser(pp, "species3", m_out_species3_name);
    }


    if (m_collision_type == "a") {
        // Initialize crosssection data for collision calculation
        m_crosssection_data.resize(10);
        for (int i=0; i<10 ; ++i) {
            m_crosssection_data[i] = i;
        }
        m_crosssection_data.copyToDeviceAsync();
    }
}

void
Collision::doCollision (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma)
{
    if (m_collision_type == "a") {
        doCollisionA(lev, bx, geom, multi_plasma);
    } else {
        doCollisionB(lev, bx, geom, multi_plasma);
    }
}

void
Collision::doCollisionA (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma)
{
    auto p_crosssection_data = m_crosssection_data.data();

    doCollisionImp(lev, bx, geom, multi_plasma,
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2,
                              amrex::RandomEngine const& engine)
        {
            // do some calculation with the two particles to see if they should collide
            // can modify the particles here
            const amrex::Real ux1 = ptd1.rdata(PlasmaIdx::ux)[i1];
            const amrex::Real ux2 = ptd2.rdata(PlasmaIdx::ux)[i2];

            const int ion_lev = ptd1.idata(PlasmaIdx::ion_lev)[i1];
            const amrex::Real crosssection = p_crosssection_data[ion_lev];

            return (crosssection + ux1) < ux2;
        },
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2, auto ptd3, int i3,
                              amrex::RandomEngine const& engine)
        {
            // initialize the new particle
            // can modify all three particles here
            const amrex::Real ux1 = ptd1.rdata(PlasmaIdx::ux)[i1];
            const amrex::Real ux2 = ptd2.rdata(PlasmaIdx::ux)[i2];
            ptd3.rdata(PlasmaIdx::ux)[i3] = ux1 + ux2;
        });
}

void
Collision::doCollisionB (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma)
{
    doCollisionImp(lev, bx, geom, multi_plasma,
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2,
                              amrex::RandomEngine const& engine)
        {
            return false;
        },
        [=] AMREX_GPU_DEVICE (auto ptd1, int i1, auto ptd2, int i2, auto ptd3, int i3,
                              amrex::RandomEngine const& engine)
        {

        });
}

template <class F, class G>
void
Collision::doCollisionImp (
        int lev, const amrex::Box& bx, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        F const& collision_function,
        G const& ionizaiton_function)
{
    // assume the two species are different for now
    auto& species1 = multi_plasma.GetPlasma(m_inout_species1_name);
    auto& species2 = multi_plasma.GetPlasma(m_inout_species2_name);

    for (PlasmaParticleIterator pti(species1); pti.isValid(); ++pti) {
        amrex::removeInvalidParticles(species1.ParticlesAt(0, pti));
        amrex::removeInvalidParticles(species2.ParticlesAt(0, pti));
    }

    PlasmaBins bins1 = findParticlesInEachTile(bx, 1, species1, geom);
    PlasmaBins bins2 = findParticlesInEachTile(bx, 1, species2, geom);

    auto offset1 = bins1.offsetsPtr();
    auto offset2 = bins2.offsetsPtr();
    auto perm1 = bins1.permutationPtr();
    auto perm2 = bins2.permutationPtr();

    const int num_cells = bins1.numBins();

    amrex::Gpu::DeviceVector<int> num_ind_pairs(num_cells, 0);
    auto p_num_ind_pairs = num_ind_pairs.dataPtr();

    const int total_ind_pairs = amrex::Scan::PrefixSum<int>(num_cells,
        [=] AMREX_GPU_DEVICE (int i) {
            int n1 = offset1[i+1] - offset1[i];
            int n2 = offset2[i+1] - offset2[i];
            return std::min(n1, n2);
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

    for (PlasmaParticleIterator pti(species1); pti.isValid(); ++pti) {

        auto& ptile1 = species1.ParticlesAt(0, pti);
        auto& ptile2 = species2.ParticlesAt(0, pti);

        auto ptd1 = ptile1.getParticleTileData();
        auto ptd2 = ptile2.getParticleTileData();

        const int np1 = ptile1.numParticles();
        const int np2 = ptile2.numParticles();

        amrex::Gpu::DeviceVector<int> flag1(np1, 0);
        amrex::Gpu::DeviceVector<int> flag2(np2, 0);

        auto p_flag1 = flag1.dataPtr();
        auto p_flag2 = flag2.dataPtr();

        amrex::ParallelForRNG(total_ind_pairs,
            [=] AMREX_GPU_DEVICE (int ipair, amrex::RandomEngine const& engine){
                const int icell = amrex::bisect(p_num_ind_pairs, 0, num_cells, ipair);

                const int offset1_start = offset1[icell];
                const int offset1_stop = offset1[icell+1];
                const int offset2_start = offset2[icell];
                const int offset2_stop = offset2[icell+1];

                const int icoll = ipair - p_num_ind_pairs[icell];

                const int n1 = offset1_stop - offset1_start;
                const int n2 = offset2_stop - offset2_start;

                int idx1 = icoll;
                int idx2 = icoll;
                while (idx1 < n1 && idx2 < n2) {
                    const int j1 = perm1[offset1_start + idx1];
                    const int j2 = perm2[offset2_start + idx2];

                    const bool make_new_particle = collision_function(
                        ptd1, j1, ptd2, j2, engine
                    );

                    if (n2 < n1) {
                        idx1 += n2;
                        if (make_new_particle) {
                            p_flag2[j2] = 1;
                        }
                    } else {
                        idx2 += n1;
                        if (make_new_particle) {
                            p_flag1[j1] = 1;
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

        auto ptd3 = ptile3.getParticleTileData();

        amrex::ParallelForRNG(total_ind_pairs,
            [=] AMREX_GPU_DEVICE (int ipair, amrex::RandomEngine const& engine){
                const int icell = amrex::bisect(p_num_ind_pairs, 0, num_cells, ipair);

                const int offset1_start = offset1[icell];
                const int offset1_stop = offset1[icell+1];
                const int offset2_start = offset2[icell];
                const int offset2_stop = offset2[icell+1];

                const int icoll = ipair - p_num_ind_pairs[icell];

                const int n1 = offset1_stop - offset1_start;
                const int n2 = offset2_stop - offset2_start;

                int idx1 = icoll;
                int idx2 = icoll;
                while (idx1 < n1 && idx2 < n2) {
                    const int j1 = perm1[offset1_start + idx1];
                    const int j2 = perm2[offset2_start + idx2];
                    const int new_part_idx = n2 < n1 ? p_flag2[j2] : p_flag1[j1];

                    if (new_part_idx >= 0) {
                        ionizaiton_function(
                            ptd1, j1,
                            ptd2, j2,
                            ptd3, old_size3 + new_part_idx,
                            engine
                        );
                    }

                    if (n2 < n1) {
                        idx1 += n2;
                    } else {
                        idx2 += n1;
                    }
                }
            }
        );

        amrex::Gpu::streamSynchronize();
    }
}
