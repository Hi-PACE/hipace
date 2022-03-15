# HiPACE++

[![Documentation Status](https://readthedocs.org/projects/hipace/badge/?version=latest)](https://hipace.readthedocs.io/en/latest/?badge=latest)
![linux](https://github.com/Hi-PACE/hipace/workflows/linux/badge.svg?branch=development&event=push)
<!-- ![macOS](https://github.com/Hi-PACE/hipace/workflows/macos/badge.svg?branch=development&event=push) -->
[![DOI (source)](https://img.shields.io/badge/DOI%20(source)-10.5281/zenodo.5358483-blue.svg)](https://doi.org/10.5281/zenodo.5358483)
[![arXiv (paper)](https://img.shields.io/badge/arXiv%20(paper)-2109.10277-blue.svg)](https://arxiv.org/abs/2109.10277)

HiPACE++ is an open-source portable GPU-capable quasistatic particle-in-cell code for wakefield acceleration written in C++.
It is a full re-writing of the legacy code [HiPACE](http://dx.doi.org/10.1088/0741-3335/56/8/084012), the Highly efficient Plasma ACcelerator Emulator.
Its main features are:
 - Multiple beams and multiple plasma species to simulation beam-driven wakefield acceleration
 - Field ionization of the plasma using the ADK model
 - Two field solver methods, the original HiPACE predictor-corrector loop and an [explicit solver](https://arxiv.org/abs/2012.00881)
 - Diagnostics compliant with the [openPMD standard](https://github.com/openPMD/openPMD-standard)
 - Read an arbitrary particle beam from file
 - more coming soon...

HiPACE++ is built on the [AMReX](https://amrex-codes.github.io) library, which provides for particle and field data structures.

Please have a look at our [documentation](https://hipace.readthedocs.io)!
