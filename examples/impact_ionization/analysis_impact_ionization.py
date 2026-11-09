#! /usr/bin/env python3

# Copyright 2026
#
# This file is part of HiPACE++.
#
# Electron-impact ionization of neutral hydrogen by thermal electrons
# (examples/impact_ionization/inputs_impact_ionization_SI).
# Compares the ionized fraction on every slice with the global model
#     dn_e/dt = k n_e n_H,  dn_H/dt = -k n_e n_H,  k = <sigma v>_Maxwellian,
# using the RBEB cross section of src/particles/collisions/ImpactIonizationSigma.H.
#
# License: BSD-3-Clause-LBNL

import argparse
import numpy as np
from scipy.constants import c, e, m_e
from scipy.integrate import quad
import read_insitu_diagnostics as diag

parser = argparse.ArgumentParser(
    description='Compare the impact-ionization rate of HiPACE++ with theory')
parser.add_argument('--insitu', dest='insitu', required=True,
                    help='Directory containing the plasma in-situ diagnostics')
# must match examples/impact_ionization/inputs_impact_ionization_SI
parser.add_argument('--electron_density', type=float, default=1e24)
parser.add_argument('--neutral_density', type=float, default=1e24)
parser.add_argument('--temperature_in_ev', type=float, default=50.)
parser.add_argument('--n_sigma', type=float, default=5.,
                    help='allowed deviation in statistical standard deviations')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# RBEB cross section (Kim et al. 2000), as in ImpactIonizationSigma.H, hydrogen data
# ---------------------------------------------------------------------------
IONIZATION_ENERGY = 13.59843449   # eV, utils/IonizationEnergiesTable.H for H
ME_C2_EV = 510998.95
A0 = 5.29177210544e-11
ALPHA = 7.2973525643e-3
NORM = 4.0 * np.pi * A0**2 * ALPHA**4


def sigma_hydrogen(E):
    """RBEB ionization cross section (m^2) of H at electron energy E (eV)"""
    ep = E / ME_C2_EV
    bp = IONIZATION_ENERGY / ME_C2_EV     # binding energy shifted to the ionization energy
    e_red = ep / bp
    if e_red <= 1.0:
        return 0.0
    betae2 = 1.0 - 1.0 / (1.0 + ep)**2
    betab2 = 1.0 - 1.0 / (1.0 + bp)**2
    s0 = NORM / (2.0 * bp * (betae2 + 2.0 * betab2))          # U = B: beta_u = beta_b
    ep2 = (1.0 / (1.0 + 0.5 * ep))**2
    a1c = (1.0 + 2.0 * ep) / (1.0 + e_red) * ep2
    a2c = (e_red - 1.0) * bp * bp * 0.5 * ep2
    a3 = np.log(betae2 / (1.0 - betae2)) - betae2 - np.log(2.0 * bp)
    return s0 * (0.5 * a3 * (1.0 - 1.0 / e_red**2) + 1.0 - 1.0 / e_red
                 + a2c - a1c * np.log(e_red))


def rate_coefficient(T):
    """k(T) = <sigma v> (m^3/s) for Maxwellian electrons at temperature T (eV):
    k = sqrt(8 e / (pi m_e)) T^(-3/2) int sigma(E) E exp(-E/T) dE"""
    I = IONIZATION_ENERGY
    integral, _ = quad(lambda E: sigma_hydrogen(E) * E * np.exp(-E / T), I, I + 60.0 * T,
                       epsabs=0.0, epsrel=1e-10)
    return np.sqrt(8.0 * e / (np.pi * m_e)) * T**-1.5 * integral


def ionized_fraction(t, ne0, nn0, T):
    """Exact solution of the global model at constant T: n_e + n_H is constant, so
    n_e(t) = N n_e0 / (n_e0 + n_H0 exp(-k N t)), N = n_e0 + n_H0
    Every new electron is one ionized H atom: the ionized fraction is (n_e - n_e0) / n_H0"""
    k = rate_coefficient(T)
    N = ne0 + nn0
    ne = N * ne0 / (ne0 + nn0 * np.exp(-k * N * t))
    return (ne - ne0) / nn0


# ---------------------------------------------------------------------------
# HiPACE++ in-situ output: one row per time step, one column per slice
# ---------------------------------------------------------------------------
elec = diag.InSituReader(f"{args.insitu}/reduced_elec.*.txt")
hyd = diag.InSituReader(f"{args.insitu}/reduced_hyd.*.txt")

# Plain sum of the macro-particle weights on every slice ("sum(w)" would contain the
# quasi-static factor gamma/psi), added over all time steps
w_elec = elec.slice_data("sum(w*(1-vz))").sum(axis=0)
w_hyd = hyd.slice_data("sum(w*(1-vz))").sum(axis=0)
n_macro = hyd.slice_data("Np").sum(axis=0)      # hydrogen macro-particles per slice, all steps

zeta = elec.zeta                                # slice centres
head = np.argmax(zeta)
# the in-situ data of a slice is recorded before that slice is ionized,
# so the head slice still has the initial electron weight
f_sim = (w_elec - w_elec[head]) / w_hyd

# The plasma enters at the head of the box and crosses one slice in dz/c:
dz = (elec.z_hi - elec.z_lo) / elec.n_slices
t = (elec.z_hi - zeta) / c - 0.5 * dz / c       # 0 at the head slice
f_th = ionized_fraction(t, args.electron_density, args.neutral_density, args.temperature_in_ev)
f_th = np.clip(f_th, 0.0, 1.0)                  # the head slice gives -1e-16 from rounding

# Statistics: every hydrogen macro-particle is ionized independently with probability f_th
sigma = np.sqrt(f_th * (1.0 - f_th) / n_macro)
checked = f_th * n_macro >= 25
deviation = np.abs(f_sim - f_th) / np.where(sigma > 0, sigma, 1.0)

tail = np.argmin(zeta)
print(f"steps: {len(elec.step)}, slices checked: {checked.sum()} of {len(zeta)}")
print(f"ionized fraction at the tail: HiPACE++ {f_sim[tail]:.4e}, "
      f"theory {f_th[tail]:.4e} +- {sigma[tail]:.1e}")
print(f"largest deviation on the checked slices: {deviation[checked].max():.2f} sigma "
      f"(allowed {args.n_sigma})")

assert checked.sum() > 0, 'No slice with enough ionization events to compare'
assert deviation[checked].max() < args.n_sigma, \
    'Impact ionization rate does not agree with theory'