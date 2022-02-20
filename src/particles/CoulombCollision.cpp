#include "CoulombCollision.H"
#include "ShuffleFisherYates.H"
#include "TileSort.H"
#include "ElasticCollisionPerez.H"

CoulombCollision::CoulombCollision(
    const std::vector<std::string>& species_names,
    std::string const collision_name)
{
    using namespace amrex::literals;
    
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
    AMREX_ALWAYS_ASSERT(lev == 0);
    using namespace amrex::literals;
    amrex::Print()<<"here\n";

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
            // uz = c (gamma - psi - com)
            const amrex::Real* const w1 = soa1.GetRealData(PlasmaIdx::w).data();

            PlasmaBins::index_type * const indices1 = bins1.permutationPtr();
            PlasmaBins::index_type const * const offsets1 = bins1.offsetsPtr();
            amrex::Real q1 = species1.GetCharge();
            amrex::Real m1 = species1.GetMass();
            // dt given by dz and psi
            const amrex::Real dV = geom.CellSize(0)*geom.CellSize(1)*geom.CellSize(2);
            
            amrex::ParallelFor(
                n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    PlasmaBins::index_type const cell_start1 = offsets1[i_cell];
                    PlasmaBins::index_type const cell_stop1  = offsets1[i_cell+1];
                    PlasmaBins::index_type const cell_half1 = (cell_start1+cell_stop1)/2;

                    // Do not collide if there is only one particle in the cell
                    if ( cell_stop1 - cell_start1 >= 2 )
                    {
                        // shuffle
                        ShuffleFisherYates(
                            indices1, cell_start1, cell_half1 );

                        // Call the function in order to perform collisions
                        ElasticCollisionPerez(
                            cell_start1, cell_half1,
                            cell_half1, cell_stop1,
                            indices1, indices1,
                            ux1, uy1, psi1, ux1, uy1, psi1, w1, w1,
                            q1, q1, m1, m1, -1.0_rt, -1.0_rt,
                            0._rt, CoulombLog, dV );
                    }
                }
                );
            count++;
        }
        AMREX_ALWAYS_ASSERT(count == 1);
/*
// Extract low-level data
int const n_cells = bins_1.numBins();
// - Species 1
auto& soa_1 = ptile_1.GetStructOfArrays();
        ParticleReal * const AMREX_RESTRICT ux_1 =
            soa_1.GetRealData(PIdx::ux).data();
        ParticleReal * const AMREX_RESTRICT uy_1 =
            soa_1.GetRealData(PIdx::uy).data();
        ParticleReal * const AMREX_RESTRICT uz_1  =
            soa_1.GetRealData(PIdx::uz).data();
        ParticleReal const * const AMREX_RESTRICT w_1 =
            soa_1.GetRealData(PIdx::w).data();
        index_type* indices_1 = bins_1.permutationPtr();
        index_type const* cell_offsets_1 = bins_1.offsetsPtr();
        Real q1 = species_1->getCharge();
        Real m1 = species_1->getMass();

        const Real dt = WarpX::GetInstance().getdt(lev);
        Geometry const& geom = WarpX::GetInstance().Geom(lev);
        const Real dV = geom.CellSize(0)*geom.CellSize(1)*geom.CellSize(2);

        // Loop over cells
*/
    } else {
        PlasmaBins bins1 = findParticlesInEachTile(lev, bx, 1, species1, geom);
        PlasmaBins bins2 = findParticlesInEachTile(lev, bx, 1, species2, geom);
    }
};
