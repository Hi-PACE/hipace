#include "CoulombCollision.H"
#include "ShuffleFisherYates.H"
#include "TileSort.H"
#include "ElasticCollisionPerez.H"
#include "utils/HipaceProfilerWrapper.H"

CoulombCollision::CoulombCollision(
    const std::vector<std::string>& species_names,
    std::string const collision_name)
{
    using namespace amrex::literals;

    // TODO: ionization level
    // DOTO: Fix dt

    // read collision species
    std::vector<std::string> collision_species;
    amrex::ParmParse pp(collision_name);
    pp.getarr("species", collision_species);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(collision_species.size() == 2,
    "Collision species must name exactly two species.");

    // default Coulomb log, if < 0, will be computed automatically
    m_CoulombLog = -1.0;
    pp.query("CoulombLog", m_CoulombLog);

    for (int i=0; i<species_names.size(); i++)
    {
        if (species_names[i] == collision_species[0]) m_species1_index = i;
        if (species_names[i] == collision_species[1]) m_species2_index = i;
    }

    m_isSameSpecies = collision_species[0] == collision_species[1] ? true : false;
}

void
CoulombCollision::doCoulombCollision (
    int lev, const amrex::Box& bx, const amrex::Geometry& geom, PlasmaParticleContainer& species1,
    PlasmaParticleContainer& species2, const bool is_same_species, const amrex::Real CoulombLog)
{
    HIPACE_PROFILE("CoulombCollision::doCoulombCollision()");
    AMREX_ALWAYS_ASSERT(lev == 0);
    using namespace amrex::literals;

    if ( is_same_species ) // species_1 == species_2
    {
        int count = 0;

        PlasmaBins bins1 = findParticlesInEachTile(lev, bx, 1, species1, geom);
        int const n_cells = bins1.numBins();
        for (PlasmaParticleIterator pti(species1, lev); pti.isValid(); ++pti) {

            auto& aos1 = pti.GetArrayOfStructs();
            const auto& pos1 = aos1.begin();
            auto& soa1 = pti.GetStructOfArrays();
            amrex::Real* const ux1 = soa1.GetRealData(PlasmaIdx::ux).data();
            amrex::Real* const uy1 = soa1.GetRealData(PlasmaIdx::uy).data();
            amrex::Real* const psi1 = soa1.GetRealData(PlasmaIdx::psi).data();
            const amrex::Real* const w1 = soa1.GetRealData(PlasmaIdx::w).data();
            PlasmaBins::index_type * const indices1 = bins1.permutationPtr();
            PlasmaBins::index_type const * const offsets1 = bins1.offsetsPtr();
            amrex::Real q1 = species1.GetCharge();
            amrex::Real m1 = species1.GetMass();

            const amrex::Real dV = geom.CellSize(0)*geom.CellSize(1)*geom.CellSize(2);
            const amrex::Real dt = geom.CellSize(2)/PhysConstSI::c;

            amrex::ParallelForRNG(
                n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    PlasmaBins::index_type const cell_start1 = offsets1[i_cell];
                    PlasmaBins::index_type const cell_stop1  = offsets1[i_cell+1];
                    PlasmaBins::index_type const cell_half1 = (cell_start1+cell_stop1)/2;

                    if ( cell_stop1 - cell_start1 <= 1 ) return;
                    // Do not collide if there is only one particle in the cell
                    // shuffle
                    ShuffleFisherYates(
                        indices1, cell_start1, cell_half1, engine );

                    // TODO: FIX DT
                    // Call the function in order to perform collisions
                    ElasticCollisionPerez(
                        cell_start1, cell_half1,
                        cell_half1, cell_stop1,
                        indices1, indices1,
                        ux1, uy1, psi1, ux1, uy1, psi1, w1, w1,
                        q1, q1, m1, m1, -1.0_rt, -1.0_rt,
                        dt, CoulombLog, dV, engine );
                }
                );
            count++;
        }
        AMREX_ALWAYS_ASSERT(count == 1);
    } else {
        PlasmaBins bins1 = findParticlesInEachTile(lev, bx, 1, species1, geom);
        PlasmaBins bins2 = findParticlesInEachTile(lev, bx, 1, species2, geom);

        int const n_cells = bins1.numBins();

        int count = 0;
        for (PlasmaParticleIterator pti(species1, lev); pti.isValid(); ++pti) {

            auto& aos1 = pti.GetArrayOfStructs();
            const auto& pos1 = aos1.begin();
            auto& soa1 = pti.GetStructOfArrays();
            amrex::Real* const ux1 = soa1.GetRealData(PlasmaIdx::ux).data();
            amrex::Real* const uy1 = soa1.GetRealData(PlasmaIdx::uy).data();
            amrex::Real* const psi1 = soa1.GetRealData(PlasmaIdx::psi).data();
            const amrex::Real* const w1 = soa1.GetRealData(PlasmaIdx::w).data();
            PlasmaBins::index_type * const indices1 = bins1.permutationPtr();
            PlasmaBins::index_type const * const offsets1 = bins1.offsetsPtr();
            amrex::Real q1 = species1.GetCharge();
            amrex::Real m1 = species1.GetMass();

            auto index = std::make_pair(pti.index(), pti.LocalTileIndex());
            // auto& ptile2 = species2.at(index);
            auto& ptile2 = species2.ParticlesAt(lev, pti.index(), pti.LocalTileIndex());
            auto& aos2 = ptile2.GetArrayOfStructs();
            const auto& pos2 = aos2.begin();
            auto& soa2 = ptile2.GetStructOfArrays();
            amrex::Real* const ux2 = soa2.GetRealData(PlasmaIdx::ux).data();
            amrex::Real* const uy2 = soa2.GetRealData(PlasmaIdx::uy).data();
            amrex::Real* const psi2= soa2.GetRealData(PlasmaIdx::psi).data();
            const amrex::Real* const w2 = soa2.GetRealData(PlasmaIdx::w).data();
            PlasmaBins::index_type * const indices2 = bins2.permutationPtr();
            PlasmaBins::index_type const * const offsets2 = bins2.offsetsPtr();
            amrex::Real q2 = species2.GetCharge();
            amrex::Real m2 = species2.GetMass();

            const amrex::Real dV = geom.CellSize(0)*geom.CellSize(1)*geom.CellSize(2);
            const amrex::Real dt = geom.CellSize(2)/PhysConstSI::c;

            // Extract particles in the tile that `mfi` points to
            // ParticleTileType& ptile_1 = species_1->ParticlesAt(lev, mfi);
            // ParticleTileType& ptile_2 = species_2->ParticlesAt(lev, mfi);
            // Loop over cells, and collide the particles in each cell

            // Loop over cells
            amrex::ParallelForRNG(
                n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    PlasmaBins::index_type const cell_start1 = offsets1[i_cell];
                    PlasmaBins::index_type const cell_stop1  = offsets1[i_cell+1];
                    // Same for species 2
                    PlasmaBins::index_type const cell_start2 = offsets2[i_cell];
                    PlasmaBins::index_type const cell_stop2  = offsets2[i_cell+1];

                    // ux from species1 can be accessed like this:
                    // ux_1[ indices_1[i] ], where i is between
                    // cell_start_1 (inclusive) and cell_start_2 (exclusive)

                    // Do not collide if one species is missing in the cell
                    if ( cell_stop1 - cell_start1 < 1 ||
                         cell_stop2 - cell_start2 < 1 ) return;
                    // shuffle
                    ShuffleFisherYates(indices1, cell_start1, cell_stop1, engine);
                    ShuffleFisherYates(indices2, cell_start2, cell_stop2, engine);

                    // TODO: FIX DT.
                    // Call the function in order to perform collisions
                    ElasticCollisionPerez(
                        cell_start1, cell_stop1, cell_start2, cell_stop2,
                        indices1, indices2,
                        ux1, uy1, psi1, ux2, uy2, psi2, w1, w2,
                        q1, q2, m1, m2, -1.0_rt, -1.0_rt,
                        dt, CoulombLog, dV, engine );
                }
                );
            count++;
        }
        AMREX_ALWAYS_ASSERT(count == 1);
    }
};
