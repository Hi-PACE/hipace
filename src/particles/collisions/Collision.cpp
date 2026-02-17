#include "Collision.H"
#include "Hipace.H"

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

    std::string name1 = "";
    getWithParser(pp, "species1", name1);
    std::string name2 = "";
    getWithParser(pp, "species2", name1);

    std::string name3 = "";
    if (m_collision_type == "a") {
        getWithParser(pp, "species3", name1);
    }

    for (int i = 0; i < plasma_species_names.size(); ++i) {
        if (plasma_species_names[i] == name1) {
            m_inout_species1_index = i;
        }
        if (plasma_species_names[i] == name2) {
            m_inout_species2_index = i;
        }
        if (plasma_species_names[i] == name3) {
            m_out_species3_index = i;
        }
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_inout_species1_index != -1,
        "Collision: Unknown species1"
    );
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_inout_species2_index != -1,
        "Collision: Unknown species2"
    );
    if (m_collision_type == "a") {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_out_species3_index != -1,
            "Collision: Unknown species3"
        );
    }
}
