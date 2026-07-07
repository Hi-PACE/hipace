#ifndef HIPACE_IMPACT_IONIZATION_SIGMA_H_
#define HIPACE_IMPACT_IONIZATION_SIGMA_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include "utils/Constants.H"

// Index mapping for reduced arrays
// Returns the starting index in the energy arrays for a given Z
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int get_energy_index(int Z)
{
    if (Z == 1)  return 0;   // H  —  1 entry  (indices 0)
    if (Z == 7)  return 1;   // N  —  7 entries (indices 1–7)
    if (Z == 2)  return 8;   // He —  2 entries (indices 8–9)
    if (Z == 18) return 10;  // Ar — 18 entries (indices 10–27)

    return 0;   // !!!
}

namespace coll_ion
{
    using namespace amrex;

    inline constexpr Real alpha = 7.2973525643e-3;
    inline constexpr Real a0    = 5.29177210544e-11; // m
    inline constexpr Real normalization = 4.0 * MathConst::pi * a0*a0 * alpha*alpha*alpha*alpha;

    // Check if the collision energy is sufficient for ionization
    AMREX_GPU_HOST_DEVICE inline
    bool impact_energy (
        Real const ux1, Real const uy1, Real const uz1, Real const g1,
        Real const ux2, Real const uy2, Real const uz2, Real const g2,
        Real const m1, Real const m2, Real const Eion_eV, Real const c2, Real const inv_c2)
    {
        using namespace amrex::literals;

        // Tolerances for numerical checks
        constexpr Real rel_tol  = 1e-10_rt;
        constexpr Real beta_tol = 1e-12_rt;

        Real const Eion  = Eion_eV * PhysConstSI::q_e;  // [J]

        // ------------------------------------------------------------------
        // Early-out if relative motion is negligible
        // ------------------------------------------------------------------
        Real const diffx = ux1 - ux2;
        Real const diffy = uy1 - uy2;
        Real const diffz = uz1 - uz2;
        Real const diffm = std::sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
        Real const summ = std::sqrt(ux1*ux1 + uy1*uy1 + uz1*uz1) + std::sqrt(ux2*ux2 + uy2*uy2 + uz2*uz2);
        // If g = u1 - u2 = 0, do not collide.
        // Or if the relative difference is less than 1.0e-10.
        if (summ <= 0.0_rt) return false;
        if (diffm / summ < rel_tol) return false;

        // ------------------------------------------------------------------
        // Initial lab-frame momenta
        // ------------------------------------------------------------------
        Real const px1 = ux1 * m1 * PhysConstSI::c;
        Real const py1 = uy1 * m1 * PhysConstSI::c;
        Real const pz1 = uz1 * m1 * PhysConstSI::c;

        Real const px2 = ux2 * m2 * PhysConstSI::c;
        Real const py2 = uy2 * m2 * PhysConstSI::c;
        Real const pz2 = uz2 * m2 * PhysConstSI::c;

        // ------------------------------------------------------------------
        // COM velocity
        // ------------------------------------------------------------------
        Real const mass_g = m1 * g1 + m2 * g2;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mass_g > 0.0_rt, "Mass is zero in ImpactIonizationSigma");

        Real const vcx    = (px1+px2) / mass_g;
        Real const vcy    = (py1+py2) / mass_g;
        Real const vcz    = (pz1+pz2) / mass_g;
        Real const vcms   = vcx*vcx + vcy*vcy + vcz*vcz;

        Real beta2 = vcms * inv_c2;
        if (beta2 < 0.0_rt) {
            if (beta2 > -beta_tol) {
                beta2 = 0.0_rt;
            } else {
                AMREX_DEVICE_PRINTF(
                    "ImpactIonizationSigma: beta < 0, beta2=%e vcms=%e\n",
                    beta2, vcms
                );
                return false;
            }
        }

        if (beta2 >= 1.0_rt - beta_tol) {
            AMREX_DEVICE_PRINTF(
                "ImpactIonizationSigma: beta too close to 1, beta2=%e\n",
                beta2
            );
            return false;
        }

        Real const gc = 1.0_rt / std::sqrt( 1.0_rt - beta2 );

        // Stable equivalent of (gc - 1)/vcms:
        // (gamma - 1)/v^2 = gamma^2 / ((gamma + 1) c^2)
        Real const boost_v2 = gc*gc / ((gc + 1.0_rt) * c2);

        // ------------------------------------------------------------------
        // COM-frame boost of the initial momenta
        // ------------------------------------------------------------------
        Real const vcDv1 = (vcx*ux1 + vcy*uy1 + vcz*uz1) * PhysConstSI::c / g1;
        Real const vcDv2 = (vcx*ux2 + vcy*uy2 + vcz*uz2) * PhysConstSI::c / g2;

        // Boost particle 1 to COM
        Real p1sx;
        Real p1sy;
        Real p1sz;
        if ( vcms > 0.0_rt )
        {
            Real const lorentz_coeff_1 =
                ( boost_v2 * vcDv1 - gc ) * m1 * g1;
            
            p1sx = px1 + vcx*lorentz_coeff_1;
            p1sy = py1 + vcy*lorentz_coeff_1;
            p1sz = pz1 + vcz*lorentz_coeff_1;
        }
        else // If vcms = 0, don't do Lorentz-transform.
        {
            p1sx = px1;
            p1sy = py1;
            p1sz = pz1;
        }

        // Boost particle 2 to COM
        Real p2sx;
        Real p2sy;
        Real p2sz;
        if (vcms > 0.0_rt) {
            Real const lorentz_coeff_2 =
                ( boost_v2 * vcDv2 - gc ) * m2 * g2;
            
            p2sx = px2 + vcx*lorentz_coeff_2;
            p2sy = py2 + vcy*lorentz_coeff_2;
            p2sz = pz2 + vcz*lorentz_coeff_2;
        } else {
            p2sx = px2;
            p2sy = py2;
            p2sz = pz2;
        }

        // ------------------------------------------------------------------
        // Ionization probability and final-state construction
        // ------------------------------------------------------------------
        Real const p1s_sq = p1sx*p1sx + p1sy*p1sy + p1sz*p1sz;
        Real const p2s_sq = p2sx*p2sx + p2sy*p2sy + p2sz*p2sz;

        Real const E1s = std::sqrt(p1s_sq*c2 + m1*m1*c2*c2);
        Real const E2s = std::sqrt(p2s_sq*c2 + m2*m2*c2*c2);

        Real const K1s = E1s - m1*c2;
        Real const K2s = E2s - m2*c2;

        // Available initial kinetic energy in the COM frame [J]
        Real const Ecm = K1s + K2s;

        // Kinetic energy available to the 3 outgoing particles in COM [J]
        Real const Eout = Ecm - Eion;
        if (Eout <= 0.0_rt)   // Energy not sufficient for ionization 
        { return false;
        } else { return true;
        }
    }
    
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
     * @return               Ionization cross section [m^2].
     */
    AMREX_GPU_HOST_DEVICE inline
    IonizationResult sigma_E(int atomic_number, int zstar, Real e, const Real* p_binding_energies, const Real* p_ionization_energies)
    {
        Real sigma = 0.0;
        Real ionization_en = 0.0;

        int n = 0;      // degeneracy counter

        int base_idx = get_energy_index(atomic_number);
        const Real me_c2 = 510998.0;        // electron rest energy [eV]

        // Convert incident energy to m_e c^2 units
        const Real ep = e / me_c2;

        // Sum over all bound electrons (k = 0 … n_bound-1)
        for (int k = 0; k < atomic_number - zstar; k++) {

            int  idx_k   = base_idx + k;
            int  idx_k_p = base_idx + (k + 1);
            
            // Add current electron to the running group
            n++;

            Real bp = p_binding_energies[idx_k] / me_c2;  // binding energy in m_e c^2 units

            // Accumulate degenerate sub-shells (same binding energy)
            if (k < atomic_number - zstar - 1 &&
                p_binding_energies[idx_k] == p_binding_energies[idx_k_p]) {
                continue;
            }

            Real e_red = ep / bp;    // reduced energy  T / B

            if (e_red > 1.0) {
                // Relativistic beta^2 factors
                Real betae2 = 1.0 - 1.0 / ((1.0 + ep) * (1.0 + ep));
                Real betab2 = 1.0 - 1.0 / ((1.0 + bp) * (1.0 + bp));
                Real betau2 = betab2;   // RBEB: U ≈ B

                // Pre-factor  s0 = (r_e / 2c) * n / [B (beta_e^2 + beta_b^2 + beta_u^2)]
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
            }
            n = 0;   // reset degeneracy for the next shell
        }
        ionization_en = p_ionization_energies[base_idx + zstar];
        return {sigma, ionization_en};
    }
}

#endif // HIPACE_IMPACT_IONIZATION_SIGMA_H_
