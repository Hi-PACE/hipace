HiPACE++
========

.. figure:: https://user-images.githubusercontent.com/26292713/144571005-b0ba2624-48ad-4293-a8c1-a19d577c44df.png
 :alt: HiPACE++

HiPACE++ is a 3D open-source portable (GPU-capable) quasi-static particle-in-cell code written in C++, available on `GitHub <https://github.com/Hi-PACE/hipace>`__.
It is a full re-writing of the DESY-LBNL legacy code `HiPACE <http://dx.doi.org/10.1088/0741-3335/56/8/084012>`__, the Highly efficient Plasma Accelerator Emulator.
Its main features are:

- Multiple beams and plasma species to simulation beam-driven wakefield acceleration
- A laser envelope solver to simulate laser-driven wakefield acceleration
- An advanced `explicit field solver <https://doi.org/10.1103/PhysRevAccelBeams.25.104603>`__ for increased accuracy
- Diagnostics compliant with the `openPMD standard <https://github.com/openPMD/openPMD-standard>`__
- Arbitrary profiles for the beams and plasma profiles
- Readers from files for the beam and laser profiles
- Adaptive time step and sub-cycling
- Additional physics (field ionization, binary collisions, temperature effects, radiation reactions)

HiPACE++ relies on the `AMReX <https://amrex-codes.github.io>`__ library, which provides for particle and field data structures.

.. note::
   There will be a virtual HiPACE++ user workshop at the 11th of July 2023 from 4pm to 7pm CET.
   Please register on the `indico webpage <https://indico.desy.de/event/40158/>`__  in case you are interested.


The documentation is structured as follows:
.. toctree::
   :maxdepth: 1

   building/building.rst
   run/parameters.rst
   run/get_started.rst
   visualize/visualization.rst
   run/chat.rst
   contributing/contributing.rst
