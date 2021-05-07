# Hipace++

![linux](https://github.com/Hi-PACE/hipace/workflows/linux/badge.svg?branch=development&event=push)
<!-- ![macOS](https://github.com/Hi-PACE/hipace/workflows/macos/badge.svg?branch=development&event=push) -->

Hipace++ is an open-source portable (GPU-capable) quasistatic particle-in-cell code written in C++.
It is a full re-writing of the DESY-LBNL legacy code [HiPACE](http://dx.doi.org/10.1088/0741-3335/56/8/084012), the Highly efficient Plasma Accelerator Emulator.
Its main features are:
 - Multiple beams and multiple plasma species to simulation beam-driven wakefield acceleration
 - Field ionization of the plasma using the ADK model
 - Two field solver methods, the original HiPACE predictor-corrector loop and an [explicit solver](https://arxiv.org/abs/2012.00881)
 - Diagnostics compliant with the [openPMD standard](https://github.com/openPMD/openPMD-standard)
 - Read an arbitrary particle beam from file
 - more coming soon...
Hipace++ relies on the [AMReX](https://amrex-codes.github.io) library, which provides for particle and field data structures.

Feel free to have a look at our documentation!