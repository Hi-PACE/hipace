Hipace++ documentation
======================

Hipace++ is an open-source portable (GPU-capable) quasistatic particle-in-cell code written in C++.
It is a full re-writing of the DESY-LBNL legacy code `HiPACE <http://dx.doi.org/10.1088/0741-3335/56/8/084012>`__, the Highly efficient Plasma Accelerator Emulator.
Its main features are:
- Multiple beams and multiple plasma species to simulation beam-driven wakefield acceleration
- Field ionization of the plasma using the ADK model
- Two field solver methods, the original HiPACE predictor-corrector loop and an `explicit solver <https://arxiv.org/abs/2012.00881>`__
- Diagnostics compliant with the `openPMD standard <https://github.com/openPMD/openPMD-standard>`__
- Read an arbitrary particle beam from file
- more coming soon...
Hipace++ relies on the `AMReX <https://amrex-codes.github.io>`__ library, which provides for particle and field data structures.

.. toctree::
   :maxdepth: 1

   building/building.rst
   run/run.rst
   run/parameters.rst
   visualize/visualization.rst
   documentation/documentation.rst
   style.rst
   contributing.rst
