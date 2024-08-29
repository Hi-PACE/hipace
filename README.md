# HiPACE++

[![Documentation Status](https://readthedocs.org/projects/hipace/badge/?version=latest)](https://hipace.readthedocs.io/en/latest/?badge=latest)
![linux](https://github.com/Hi-PACE/hipace/workflows/linux/badge.svg?branch=development&event=push)
<!-- ![macOS](https://github.com/Hi-PACE/hipace/workflows/macos/badge.svg?branch=development&event=push) -->
[![DOI (source)](https://img.shields.io/badge/DOI%20(source)-10.5281/zenodo.5358483-blue.svg)](https://doi.org/10.5281/zenodo.5358483)
[![DOI (paper)](https://img.shields.io/badge/DOI%20(paper)-10.1016/j.cpc.2022.108421-blue.svg)](https://doi.org/10.1016/j.cpc.2022.108421)

HiPACE++ is an open-source portable GPU-capable quasi-static particle-in-cell code for wakefield acceleration written in C++.
It is a full re-writing of the legacy code [HiPACE](http://dx.doi.org/10.1088/0741-3335/56/8/084012), the Highly efficient Plasma ACcelerator Emulator.
Its main features are:
 - Multiple beam and plasma species to simulation beam-driven wakefield acceleration
 - A laser envelope solver to simulate laser-driven wakefield acceleration
 - An advanced [explicit field solver](https://doi.org/10.1103/PhysRevAccelBeams.25.104603) for increased accuracy
 - Diagnostics compliant with the [openPMD standard](https://github.com/openPMD/openPMD-standard)
 - Arbitrary profiles for the beams and plasma profiles
 - Readers from files for the beam and laser profiles
 - Adaptive time step and sub-cycling
 - Additional physics (field ionization, binary collisions, temperature effects, radiation reactions)

HiPACE++ is built on the [AMReX](https://amrex-codes.github.io) library, which provides for particle and field data structures.

Please have a look at our [documentation](https://hipace.readthedocs.io) and join the [chat](https://hipace.readthedocs.io/en/latest/run/chat.html)!

## Copyright Notice

HiPACE++ Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and Deutsches
Elektronen-Synchrotron (DESY). All rights reserved.

Please see the full license agreement and notices in [license.txt](license.txt).  
Please see the notices in [legal.txt](legal.txt).  
The SPDX license identifier is `BSD-3-Clause-LBNL`.
