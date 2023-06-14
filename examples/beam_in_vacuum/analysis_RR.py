#! /usr/bin/env python3

# Copyright 2020-2023
#
# This file is part of HiPACE++. It tests the radiation reaction of a beam in an external field vs
# the analytic theory in P. Michel et al., PRE 74, 026501 https://doi.org/10.1103/PhysRevE.74.026501
#
# Authors: Severin Diederichs
# License: BSD-3-Clause-LBNL


import numpy as np
import sys
sys.path.append("../../tools/")
import read_insitu_diagnostics as diag
from scipy import constants as scc

# load HiPACE++ data with insitu diags
all_data_with_RR = diag.read_file('diags/insitu/reduced_beam.*.txt')

ne = 5e24
wp = np.sqrt(ne*scc.e**2 / (scc.m_e * scc.epsilon_0 ))
kp = wp/scc.c

mean_gamma0 = diag.gamma_mean(all_data_with_RR["average"])[0] # should be 2000
std_gamma0 = diag.gamma_spread(all_data_with_RR["average"])[0] \
            /diag.gamma_mean(all_data_with_RR["average"])[0] # should be 0.01
epsilonx0 = diag.emittance_x(all_data_with_RR["average"])[0] # sigma_x0**2 *np.sqrt(mean_gamma0/2) *kp
# should be 313e-6

# final simulation values
mean_gamma_sim = diag.gamma_mean(all_data_with_RR["average"])[-1]
std_gamma_sim = diag.gamma_spread(all_data_with_RR["average"])[-1] \
            /diag.gamma_mean(all_data_with_RR["average"])[-1]
epsilonx_sim = diag.emittance_x(all_data_with_RR["average"])[-1]

# calculate theoretical values
sigma_x0 = np.sqrt(epsilonx0 / (kp* np.sqrt(mean_gamma0/2))) # should be 4.86e-6
ux0 = epsilonx0/sigma_x0
r_e = 1/(4*np.pi*scc.epsilon_0) * scc.e**2/(scc.m_e*scc.c**2)
taur = 6.24e-24 # 2*r_e /(3*scc.c)
K = kp/np.sqrt(2)
w_beta = K*scc.c/np.sqrt(mean_gamma0)
xmsq = sigma_x0**2 + scc.c**2*ux0**2/(w_beta**2 * mean_gamma0**2)
nugamma = taur * scc.c**2 * K**4 * mean_gamma0 * xmsq/2 # equation 32 from the paper
nugammastd = taur * scc.c**2 * K**4 * mean_gamma0 * sigma_x0**2

t = all_data_with_RR["time"][-1]
gamma_theo = mean_gamma0/(1+nugamma*t) # equation 31 from the paper
std_gamma_theo = np.sqrt(std_gamma0**2 + nugammastd**2 * t**2) # equation 35 from the paper
emittance_theo = epsilonx0/(1+3*nugammastd*t/2) # equation 39 from the paper

error_analytic_gamma = np.abs(mean_gamma_sim-gamma_theo)/gamma_theo
error_analytic_std_gamma = np.abs(std_gamma_sim-std_gamma_theo)/std_gamma_theo
error_analytic_emittance = np.abs(epsilonx_sim-emittance_theo)/emittance_theo

print("Error on gamma ", error_analytic_gamma)
print("Error on relative gamma spread ", error_analytic_std_gamma)
print("Error on emittance ", error_analytic_emittance)
assert(error_analytic_gamma < 1e-3)
assert(error_analytic_std_gamma < 3e-2)
assert(error_analytic_emittance < 1e-3)
