#! /usr/bin/env python3

# Copyright 2026
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

# This script compares the Courant-Snyder parameters of the beam with the theoretical value,
# and asserts that the difference is small.

import numpy as np
import argparse
from openpmd_viewer.addons import LpaDiagnostics

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

ts = LpaDiagnostics(args.output_dir)

def get_beam_parameters(iteration):
    x, y, ux, uy, uz = ts.get_particle(["x", "y", "ux", "uy", "uz"], iteration=iteration)

    x_prime = ux / uz
    y_prime = uy / uz

    mean_x = np.mean(x)
    mean_xp = np.mean(x_prime)
    mean_y = np.mean(y)
    mean_yp = np.mean(y_prime)

    var_x = np.var(x)
    var_xp = np.var(x_prime)
    cov_xxp = np.mean((x - mean_x) * (x_prime - mean_xp))

    var_y = np.var(y)
    var_yp = np.var(y_prime)
    cov_yyp = np.mean((y - mean_y) * (y_prime - mean_yp))

    emit_x = np.sqrt(var_x * var_xp - cov_xxp**2)
    beta_x  = var_x / emit_x
    alpha_x = -cov_xxp / emit_x
    gamma_x = (1 + alpha_x**2) / beta_x

    emit_y = np.sqrt(var_y * var_yp - cov_yyp**2)
    beta_y  = var_y / emit_y
    alpha_y = -cov_yyp / emit_y
    gamma_y = (1 + alpha_y**2) / beta_y

    p0 = np.mean(uz)
    emit_nx = emit_x * p0
    emit_ny = emit_y * p0

    lpa_diag_energy = ts.get_energy_spread(iteration=iteration, property='energy')
    lpa_diag_charge = ts.get_charge(iteration=iteration)
    lpa_diag_divergence = ts.get_divergence(iteration=iteration)
    lpa_diag_emittance = ts.get_emittance(iteration=iteration)

    return [
        lpa_diag_energy[0],
        lpa_diag_energy[1],
        lpa_diag_charge,
        lpa_diag_divergence[0],
        lpa_diag_divergence[1],
        lpa_diag_emittance[0],
        lpa_diag_emittance[1],
        alpha_x,
        beta_x,
        gamma_x,
        emit_nx,
        alpha_y,
        beta_y,
        gamma_y,
        emit_ny
    ]


def evolve_twiss(params):
    L = 200e-6
    new_params = params.copy()

    new_params[8] += - new_params[7] * 2 * L + new_params[9] * L**2 # beta_x
    new_params[12] += - new_params[11] * 2 * L + new_params[13] * L**2 # beta_y
    new_params[7] -= new_params[9] * L # alpha_x
    new_params[11] -= new_params[13] * L # alpha_y

    return new_params


initial_beam_params = [
    1e3,
    10,
    -3e-10,
    0.00135,
    0.00140,
    1.2e-6,
    1.3e-6,
    1.4,
    1e-3,
    (1 + 1.4**2) / 1e-3,
    1.2e-6,
    -1.5,
    1.1e-3,
    (1 + 1.5**2) / 1.1e-3,
    1.3e-6
]

for i in ts.iterations:
    params = get_beam_parameters(i)
    max_rel_diff = max(abs((p - r) / r) for p, r in zip(params, initial_beam_params))

    print("iteration", i)
    print("analytic parameters", initial_beam_params)
    print("hipace beam parameters", params)
    print("max relative error", max_rel_diff)
    assert max_rel_diff < 0.01

    initial_beam_params = evolve_twiss(initial_beam_params)
