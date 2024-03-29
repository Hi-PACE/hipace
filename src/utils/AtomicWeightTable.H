/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
// This script was automatically generated!
// Edit src/utils/write_atomic_weight_cpp.py instead!
#ifndef HIPACE_ATOMIC_WEIGHT_TABLE_H_
#define HIPACE_ATOMIC_WEIGHT_TABLE_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>

// Reference:
// Coursey, J.S., Schwab, D.J., Tsai, J.J., and Dragoset, R.A. (2015),
// Atomic Weights and Isotopic Compositions (version 4.1).
// [Online] Available: http://physics.nist.gov/Comp [2021, 05, 19].
// National Institute of Standards and Technology, Gaithersburg, MD.
//
// The Data written below is a reformatting of the data referenced form NIST.

std::map<std::string, amrex::Real> standard_atomic_weights = {
    {"H", 1.007975},
    {"D", 2.014101778},
    {"T", 3.016049278},
    {"He", 4.002602},
    {"Li", 6.9675},
    {"Be", 9.0121831},
    {"B", 10.8135},
    {"C", 12.0106},
    {"N", 14.006855},
    {"O", 15.9994},
    {"F", 18.998403163},
    {"Ne", 20.1797},
    {"Na", 22.98976928},
    {"Mg", 24.3055},
    {"Al", 26.9815385},
    {"Si", 28.085},
    {"P", 30.973761998},
    {"S", 32.0675},
    {"Cl", 35.4515},
    {"Ar", 39.948},
    {"K", 39.0983},
    {"Ca", 40.078},
    {"Sc", 44.955908},
    {"Ti", 47.867},
    {"V", 50.9415},
    {"Cr", 51.9961},
    {"Mn", 54.938044},
    {"Fe", 55.845},
    {"Co", 58.933194},
    {"Ni", 58.6934},
    {"Cu", 63.546},
    {"Zn", 65.38},
    {"Ga", 69.723},
    {"Ge", 72.63},
    {"As", 74.921595},
    {"Se", 78.971},
    {"Br", 79.904},
    {"Kr", 83.798},
    {"Rb", 85.4678},
    {"Sr", 87.62},
    {"Y", 88.90584},
    {"Zr", 91.224},
    {"Nb", 92.90637},
    {"Mo", 95.95},
    {"Tc", 98.0},
    {"Ru", 101.07},
    {"Rh", 102.9055},
    {"Pd", 106.42},
    {"Ag", 107.8682},
    {"Cd", 112.414},
    {"In", 114.818},
    {"Sn", 118.71},
    {"Sb", 121.76},
    {"Te", 127.6},
    {"I", 126.90447},
    {"Xe", 131.293},
    {"Cs", 132.90545196},
    {"Ba", 137.327},
    {"La", 138.90547},
    {"Ce", 140.116},
    {"Pr", 140.90766},
    {"Nd", 144.242},
    {"Pm", 145.0},
    {"Sm", 150.36},
    {"Eu", 151.964},
    {"Gd", 157.25},
    {"Tb", 158.92535},
    {"Dy", 162.5},
    {"Ho", 164.93033},
    {"Er", 167.259},
    {"Tm", 168.93422},
    {"Yb", 173.054},
    {"Lu", 174.9668},
    {"Hf", 178.49},
    {"Ta", 180.94788},
    {"W", 183.84},
    {"Re", 186.207},
    {"Os", 190.23},
    {"Ir", 192.217},
    {"Pt", 195.084},
    {"Au", 196.966569},
    {"Hg", 200.592},
    {"Tl", 204.3835},
    {"Pb", 207.2},
    {"Bi", 208.9804},
    {"Po", 209.0},
    {"At", 210.0},
    {"Rn", 222.0},
    {"Fr", 223.0},
    {"Ra", 226.0},
    {"Ac", 227.0},
    {"Th", 232.0377},
    {"Pa", 231.03588},
    {"U", 238.02891},
    {"Np", 237.0},
    {"Pu", 244.0}, };

#endif // #ifndef ATOMIC_WEIGHT_TABLE_H_
