#ifndef HIPACE_IMPACT_IONIZATION_SIGMA_H_
#define HIPACE_IMPACT_IONIZATION_SIGMA_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
// #include <map>
#include "utils/Constants.H"

// Quantum degeneracy for each species (g = 2J + 1, ground-state term)
/*
AMREX_GPU_HOST_DEVICE inline amrex::Real g_deg[29] = {
    // H (2 stages)
    2.0, 1.0,
    // N (6 stages)
    4.0, 9.0, 6.0, 1.0, 2.0, 1.0,
    // He (3 stages)
    1.0, 2.0, 1.0,
    // Ar (18 stages)
    1.0, 4.0, 5.0, 4.0, 9.0, 6.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0
};

constexpr int num_atom_species = 4;

constexpr amrex::Real X_min = 1e-20; // Threshold for considering a species to be present

// Helper lambda to map species index to atomic species
AMREX_GPU_HOST_DEVICE
inline int get_atom_index(int mu) {
    if (mu <= 1 && mu >= 0) return 0;     // Hydrogen
    if (mu <= 7 && mu > 1) return 1;      // Nitrogen
    if (mu <= 10 && mu > 7) return 2;     // Helium
    if (mu <= 14 && mu > 10) return 3;    // Argon
    return 0;                             //!!!
};

AMREX_GPU_HOST_DEVICE
const int zatom[num_atom_species] = {1, 7, 2, 18}; // Atomic numbers for H, N, He, Ar
*/

// Index mapping for reduced arrays
// Returns the starting index in the energy arrays for a given Z
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int get_energy_index(int Z)
{
    if (Z == 1)  return 0;   // H  —  1 entry  (indices 0)
    if (Z == 7)  return 1;   // N  —  7 entries (indices 1–7)
    if (Z == 2)  return 8;   // He —  2 entries (indices 8–9)
    if (Z == 18) return 10;  // Ar — 18 entries (indices 10–27)

    return 0;   //!!!
}

namespace coll_ion
{
    using namespace amrex;

    inline constexpr Real alpha = 7.2973525643e-3;
    inline constexpr Real a0    = 5.29177210544e-11; // m
    inline constexpr Real normalization = 4.0 * MathConst::pi * a0*a0 * alpha*alpha*alpha*alpha;

    struct IonizationResult
    {
        Real sigma;
        Real ionization_en;
    };

    // Cross section using RBEB model from Kim et al. (2000)
    /**
     * @brief Relativistic Binary-Encounter-Bethe (RBEB) electron-impact
     *        ionization cross section.
     *
     * Loops over all bound electrons of the ion (from innermost to the
     * outermost occupied shell).  Degenerate sub-shells sharing the same
     * binding energy are grouped and counted together (factor `n`).
     *
     * @param atomic_number  Atomic number Z of the target element.
     * @param zstar          Current ion charge (number of removed electrons).
     * @param e              Incident-electron kinetic energy [eV].
     * @return               Ionization cross section [m²].
     */
    AMREX_GPU_HOST_DEVICE inline
    IonizationResult sigma_E(int atomic_number, int zstar, Real e, const Real* p_binding_energies, const Real* p_ionization_energies)
    {
        Real sigma = 0.0;
        Real weighted_en = 0.0;

        Real ionization_sum = 0.0;
        int n = 0;      // degeneracy counter

        int base_idx = get_energy_index(atomic_number);
        const Real me_c2 = 510998.0;        // electron rest energy [eV]

        // Convert incident energy to m_e c² units
        const Real ep = e / me_c2;

        // Sum over all bound electrons (k = 0 … n_bound-1)
        for (int k = 0; k < atomic_number - zstar; k++) {

            int  idx_k   = base_idx + k;
            int  idx_k_p = base_idx + (k + 1);
            
            // Add current electron to the running group
            n++;
            ionization_sum += p_ionization_energies[idx_k];

            Real bp = p_binding_energies[idx_k] / me_c2;  // binding energy in m_e c² units

            // Accumulate degenerate sub-shells (same binding energy)
            if (k < atomic_number - zstar - 1 &&
                p_binding_energies[idx_k] == p_binding_energies[idx_k_p]) {
                continue;
            }

            Real e_red = ep / bp;    // reduced energy  T / B

            if (e_red > 1.0) {
                // Relativistic β² factors
                Real betae2 = 1.0 - 1.0 / ((1.0 + ep) * (1.0 + ep));
                Real betab2 = 1.0 - 1.0 / ((1.0 + bp) * (1.0 + bp));
                Real betau2 = betab2;   // RBEB: U ≈ B

                // Pre-factor  s0 = (r_e / 2c) * n / [B (βe² + βb² + βu²)]
                Real s0 = normalization * n /
                          (2.0 * bp * (betae2 + betab2 + betau2));

                Real ep2 = 1.0 / (1.0 + 0.5 * ep);
                ep2 *= ep2;

                // RBEB kernel terms
                Real a1c = (1.0 + 2.0 * ep) / (1.0 + e_red) * ep2;
                Real a2c = (e_red - 1.0) * bp * bp * 0.5 * ep2;
                Real a3  = std::log(betae2 / (1.0 - betae2))
                         - betae2
                         - std::log(2.0 * bp);

                Real sk = s0 * (0.5 * a3 * (1.0 - 1.0 / (e_red * e_red))
                              + 1.0 - 1.0 / e_red
                              + a2c
                              - a1c * std::log(e_red));

                sigma += sk;

                //is this correct? should degenerate electrons share one averaged ionization energy?
                Real ionization_E = ionization_sum / static_cast<Real>(n);
                weighted_en += sk * ionization_E;
            }

            n = 0;   // reset degeneracy for the next shell
            ionization_sum = 0.0;
        }
        Real ionization_en = (sigma > 0.0) ? weighted_en / sigma : 0.0;

        return {sigma, ionization_en};
    }
}

#endif // HIPACE_IMPACT_IONIZATION_SIGMA_H_
