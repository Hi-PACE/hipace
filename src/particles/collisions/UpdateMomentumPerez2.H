/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef HIPACE_UPDATE_MOMENTUM_PEREZ_ELASTIC2_H_
#define HIPACE_UPDATE_MOMENTUM_PEREZ_ELASTIC2_H_

#include "utils/Constants.H"

#include <AMReX_Random.H>

#include <cmath>  // isnan() isinf()
#include <limits> // numeric_limits<float>::min()

/* \brief Update particle velocities according to
 *        F. Perez et al., Phys.Plasmas.19.083104 (2012),
 *        which is based on Nanbu's method, PhysRevE.55.4642 (1997).
 *        @param[in] LmdD is max(Debye length, minimal interparticle distance).
 *        @param[in] L is the Coulomb log. A fixed L will be used if L > 0,
 *        otherwise L will be calculated based on the algorithm.
 *        To see if there are nan or inf updated velocities,
 *        compile with USE_ASSERTION=TRUE.
*/
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumPerezElastic2 (
    amrex::Real& u1x, amrex::Real& u1y, amrex::Real& u1z, const amrex::Real g1,
    amrex::Real& u2x, amrex::Real& u2y, amrex::Real& u2z, const amrex::Real g2,
    const amrex::Real n1, const amrex::Real n2, const amrex::Real n12,
    const amrex::Real q1, amrex::Real m1, const amrex::Real w1,
    const amrex::Real q2, amrex::Real m2, const amrex::Real w2,
    const amrex::Real dt, const amrex::Real L,  const amrex::Real lmdD,
    const amrex::Real inv_c_SI, const amrex::Real inv_c2_SI, const bool normalized_units,
    amrex::RandomEngine const& engine)
{
    using namespace amrex::literals;

    const amrex::Real diffx = amrex::Math::abs(u1x - u2x);
    const amrex::Real diffy = amrex::Math::abs(u1y - u2y);
    const amrex::Real diffz = amrex::Math::abs(u1z - u2z);
    const amrex::Real diffm = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);
    const amrex::Real summm = std::sqrt(u1x * u1x + u1y * u1y + u1z * u1z) +
                              std::sqrt(u2x * u2x + u2y * u2y + u2z * u2z);
    // If g = u1 - u2 = 0, do not collide.
    // Or if the relative difference is less than 1.0e-10.
    if (diffm < std::numeric_limits<amrex::Real>::min() || diffm < 1.0e-10 * summm) {
        return;
    }

    // Transform to SI units (u is the normalized momentum here)
    if (normalized_units) {
        m1 *= PhysConstSI::m_e;
        m2 *= PhysConstSI::m_e;
    }
    // Compute momenta
    const amrex::Real p1x = u1x * m1 * PhysConstSI::c;
    const amrex::Real p1y = u1y * m1 * PhysConstSI::c;
    const amrex::Real p1z = u1z * m1 * PhysConstSI::c;
    const amrex::Real p2x = u2x * m2 * PhysConstSI::c;
    const amrex::Real p2y = u2y * m2 * PhysConstSI::c;
    const amrex::Real p2z = u2z * m2 * PhysConstSI::c;

    // Compute center-of-mass (COM) velocity and gamma
    const amrex::Real mass_g = m1 * g1 + m2 * g2;
    const amrex::Real mass_g_inv = 1._rt / mass_g;
    const amrex::Real vcx = (p1x + p2x) * mass_g_inv;
    const amrex::Real vcy = (p1y + p2y) * mass_g_inv;
    const amrex::Real vcz = (p1z + p2z) * mass_g_inv;
    const amrex::Real vcms = vcx * vcx + vcy * vcy + vcz * vcz;
    const amrex::Real gc_inv = std::sqrt(std::max(1._rt - vcms * inv_c2_SI, 0._rt));
    if (gc_inv == 0._rt) {
        return;
    }
    const amrex::Real gc = 1._rt / gc_inv;

    // Compute vc dot v1 and v2
    const amrex::Real vcDv1 = (vcx * u1x + vcy * u1y + vcz * u1z) * PhysConstSI::c / g1;
    const amrex::Real vcDv2 = (vcx * u2x + vcy * u2y + vcz * u2z) * PhysConstSI::c / g2;

    // Compute p1 star
    amrex::Real p1sx = p1x;
    amrex::Real p1sy = p1y;
    amrex::Real p1sz = p1z;
    // If vcms = 0, don't do Lorentz-transform.
    if (vcms > std::numeric_limits<amrex::Real>::min()) {
        const amrex::Real lorentz_transform_factor = ((gc - 1._rt) / vcms * vcDv1 - gc) * m1 * g1;
        p1sx += vcx * lorentz_transform_factor;
        p1sy += vcy * lorentz_transform_factor;
        p1sz += vcz * lorentz_transform_factor;
    }
    const double p1sm = std::sqrt(static_cast<double>(p1sx) * p1sx + p1sy * p1sy + p1sz * p1sz);
    const double p1sm_inv_sq = 1. / (p1sm * p1sm * inv_c2_SI);

    // Compute gamma star
    const amrex::Real g1s = (1._rt - vcDv1 * inv_c2_SI) * gc * g1;
    const amrex::Real g2s = (1._rt - vcDv2 * inv_c2_SI) * gc * g2;

    // Compute the Coulomb log lnLmd
    double lnLmd = L;
    if (lnLmd <= 0._rt) {
        // Compute b0 according to eq (22) from Perez et al.,
        // Phys.Plasmas.19.083104 (2012) Note: there is a typo in the equation,
        // the last square is incorrect! See the SMILEI documentation:
        // https://smileipic.github.io/Smilei/Understand/collisions.html
        const double b0 = amrex::Math::abs(q1 * q2) * inv_c2_SI *
            (1._rt / (4._rt * MathConst::pi * PhysConstSI::ep0))
            * gc * mass_g_inv * (m1 * g1s * m2 * g2s * p1sm_inv_sq + 1._rt);

        // Compute the minimal impact parameter
        double bmin_inv = 1. / std::max(PhysConstSI::hbar * MathConst::pi * (1. / p1sm), b0);

        // Compute the Coulomb log lnLmd according to eq (23) from Perez et al.,
        // Phys.Plasmas.19.083104 (2012)
        lnLmd = amrex::max(static_cast<double>(2.0),
            0.5_rt * std::log(1._rt + lmdD * lmdD * bmin_inv * bmin_inv));
    }

    // Compute s according to eq (17) from Perez et al., Phys.Plasmas.19.083104
    // (2012)
    const double tts = static_cast<double>(m1) * g1s * m2 * g2s * p1sm_inv_sq + 1._rt;
    const double tts2 = tts * tts;
    const double charge_fac = normalized_units ? static_cast<double>(PhysConstSI::q_e)
        * PhysConstSI::q_e * PhysConstSI::q_e * PhysConstSI::q_e : 1.;
    amrex::Real s = static_cast<double>(n1) * n2 / n12 * dt * lnLmd * q1 * q1 *
        q2 * q2 * charge_fac * inv_c2_SI * inv_c2_SI / (static_cast<double>(4.0) * MathConst::pi *
        PhysConstSI::ep0 * PhysConstSI::ep0 * m1 * g1 * m2 * g2) * gc * p1sm * mass_g_inv * tts2;

    const auto cbrt_n1 = std::cbrt(n1);
    const auto cbrt_n2 = std::cbrt(n2);
    const auto coeff = static_cast<amrex::Real>(std::pow(4.0 * MathConst::pi / 3.0, 1.0 / 3.0));
    const amrex::Real vrel = static_cast<double>(mass_g) * p1sm / (m1 * g1s * m2 * g2s * gc);
    const amrex::Real sp = static_cast<double>(coeff) * n1 * n2 / n12 * dt * vrel * (m1 + m2) /
        amrex::max(m1 * cbrt_n1 * cbrt_n1, m2 * cbrt_n2 * cbrt_n2);

    // Determine s
    s = amrex::min(s, sp);

    // Get random numbers
    amrex::Real r = amrex::Random(engine);

    // Compute scattering angle according to eq (10) and following from Perez et
    // al., Phys.Plasmas.19.083104 (2012)
    amrex::Real cosXs = 0;
    if (s <= 0.1_rt) {
        while (true) {
            cosXs = 1._rt + s * std::log(r);
            // Avoid the bug when r is too small such that cosXs < -1
            if (cosXs >= -1._rt) {
                break;
            }
            r = amrex::Random(engine);
        }
    } else if (s > 0.1_rt && s <= 3._rt) {
        const amrex::Real Ainv = static_cast<amrex::Real>(
            0.0056958_rt
            + 0.9560202_rt * s
            - 0.508139_rt * s * s
            + 0.47913906_rt * s * s * s
            - 0.12788975_rt * s * s * s * s
            + 0.02389567_rt * s * s * s * s * s);
        cosXs = Ainv * std::log(std::exp(-(1._rt / Ainv)) + 2._rt * r * std::sinh(1._rt / Ainv));
    } else if (s > 3._rt && s <= 6._rt) {
        const amrex::Real A = 3._rt * std::exp(-s);
        cosXs = 1._rt / A * std::log(std::exp(-A) + 2._rt * r * std::sinh(A));
    } else {
        cosXs = 2._rt * r - 1._rt;
    }
    const amrex::Real sinXs = std::sqrt(1._rt - cosXs * cosXs);

    // Get random azimuthal angle
    const amrex::Real phis = amrex::Random(engine) * 2._rt * MathConst::pi;
    const amrex::Real cosphis = std::cos(phis);
    const amrex::Real sinphis = std::sin(phis);

    // Compute post-collision momenta pfs in COM
    amrex::Real p1fsx = 0;
    amrex::Real p1fsy = 0;
    amrex::Real p1fsz = 0;
    // p1sp is the p1s perpendicular
    double p1sp = std::sqrt(static_cast<double>(p1sx) * p1sx + p1sy * p1sy);
    // Make sure p1sp is not almost zero
    if (p1sp > std::numeric_limits<amrex::Real>::min()) {
        const double p1sp_inv = 1. / p1sp;
        p1fsx = (p1sx * p1sz * p1sp_inv) * sinXs * cosphis +
                (p1sy * p1sm * p1sp_inv) * sinXs * sinphis + (p1sx)*cosXs;
        p1fsy = (p1sy * p1sz * p1sp_inv) * sinXs * cosphis +
                (-p1sx * p1sm * p1sp_inv) * sinXs * sinphis + (p1sy)*cosXs;
        p1fsz = (-p1sp) * sinXs * cosphis +
                (0._rt) * sinXs * sinphis + (p1sz)*cosXs;
        // Note a negative sign is different from
        // Eq. (12) in Perez's paper,
        // but they are the same due to the random nature of phis.
    } else {
        // If the previous p1sp is almost zero
        // x->y  y->z  z->x
        // This set is equivalent to the one in Nanbu's paper
        p1sp = std::sqrt(static_cast<double>(p1sy) * p1sy + p1sz * p1sz);
        const double p1sp_inv = 1. / p1sp;
        p1fsy = (p1sy * p1sx * p1sp_inv) * sinXs * cosphis +
                (p1sz * p1sm * p1sp_inv) * sinXs * sinphis + (p1sy)*cosXs;
        p1fsz = (p1sz * p1sx * p1sp_inv) * sinXs * cosphis +
                (-p1sy * p1sm * p1sp_inv) * sinXs * sinphis + (p1sz)*cosXs;
        p1fsx = (-p1sp) * sinXs * cosphis +
                (0._rt) * sinXs * sinphis + (p1sx)*cosXs;
    }

    const amrex::Real p2fsx = -p1fsx;
    const amrex::Real p2fsy = -p1fsy;
    const amrex::Real p2fsz = -p1fsz;

    // Transform from COM to lab frame
    amrex::Real p1fx = p1fsx;
    amrex::Real p2fx = p1fsy;
    amrex::Real p1fy = p1fsz;
    amrex::Real p2fy = p2fsx;
    amrex::Real p1fz = p2fsy;
    amrex::Real p2fz = p2fsz;
    // If vcms = 0, don't do Lorentz-transform.
    if (vcms > std::numeric_limits<amrex::Real>::min()) {
        const amrex::Real vcDp1fs = vcx * p1fsx + vcy * p1fsy + vcz * p1fsz;
        const amrex::Real vcDp2fs = vcx * p2fsx + vcy * p2fsy + vcz * p2fsz;
        const amrex::Real factor = (gc - 1._rt) / vcms;
        const amrex::Real factor1 = factor * vcDp1fs + m1 * g1s * gc;
        const amrex::Real factor2 = factor * vcDp2fs + m2 * g2s * gc;
        p1fx += vcx * factor1;
        p1fy += vcy * factor1;
        p1fz += vcz * factor1;
        p2fx += vcx * factor2;
        p2fy += vcy * factor2;
        p2fz += vcz * factor2;
    }

    // Rejection method according to eq (14) and following from Perez et al.,
    // Phys.Plasmas.19.083104 (2012)
    const amrex::Real m1_inv = 1._rt / m1;
    const amrex::Real m2_inv = 1._rt / m2;
    if (w2 > amrex::Random(engine) * amrex::max(w1, w2)) {
        u1x = inv_c_SI * p1fx * m1_inv;
        u1y = inv_c_SI * p1fy * m1_inv;
        u1z = inv_c_SI * p1fz * m1_inv;
    }
    if (w1 > amrex::Random(engine) * amrex::max(w1, w2)) {
        u2x = inv_c_SI * p2fx * m2_inv;
        u2y = inv_c_SI * p2fy * m2_inv;
        u2z = inv_c_SI * p2fz * m2_inv;
    }
}

#endif // HIPACE_UPDATE_MOMENTUM_PEREZ_ELASTIC2_H_
