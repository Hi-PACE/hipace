.. _install-hpc:

HPC platforms
=============

Specific installation instructions for a set of supercomputers can be found below.
Follow the guide here instead of the generic installation routines for optimal stability and best performance.


.. _install-hpc-profile:

profile.hipace
--------------

Use a ``profile.hipace`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   source $HOME/profile.hipace

We list example ``profile.hipace`` files below, which can be used to set up HiPACE++ on various HPC systems.


.. _install-hpc-machines:

HPC Machines
------------

This section documents quick-start guides for a selection of supercomputers that HiPACE++ users are active on.

.. toctree::
   :maxdepth: 1

   platforms/booster_jsc
   platforms/lumi_csc
   platforms/maxwell_desy
   platforms/perlmutter_nersc

.. tip::

   Your HPC system is not in the list?
   `Open an issue <https://github.com/Hi-PACE/hipace/issues>`__ and we can document it together!
