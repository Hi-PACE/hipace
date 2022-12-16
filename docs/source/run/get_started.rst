Get started
===========

A full HiPACE++ input file can be found below, with a detailed description of the main parameters.
All the input parameters available in the simulation are available in the :doc:`parameters` section.

.. literalinclude:: ../../../examples/get_started/inputs_normalized

This input file is ``inputs_normalized`` in ``examples/get_started/``, which is the recommended place to start running a first simulation.

See the :doc:`../building/building` section to install the code locally or on your favorite platform.
After compiling HiPACE++ from the HiPACE++ root directory, execute

.. code-block:: bash

   cd examples/get_started/
   ../../build/bin/hipace inputs_normalized # run the simulation

Then you can use the `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`__ to read the simulation output data. Directory `examples/get_started/` also contains this :download:`example notebook <../../../examples/get_started/notebook.ipynb>`. Reading the simulation output only takes these few Python commands:

.. code-block:: python

   # import statements
   import numpy as np
   import matplotlib.pyplot as plt
   from openpmd_viewer import OpenPMDTimeSeries
   # Read the simulation data
   ts = OpenPMDTimeSeries('./diags/hdf5/')
   # Get beam and field data at iteration 20
   iteration = 20
   x, z = ts.get_particle(species='beam', iteration=iteration, var_list=['x', 'z'])
   F, m = ts.get_field(field='Ez', iteration=iteration)

Also please have a look at the production runs for beam-driven and laser-driven wakefield:

.. toctree::
    :maxdepth: 1

    inputs_pwfa.rst
    inputs_lwfa.rst

Both use realistic resolutions and run well on modern GPUs (as an estimate, each should take less than 5 min on 16 modern GPUs, e.g., nvidia A100).

Additional examples can be found in ``examples/``.
