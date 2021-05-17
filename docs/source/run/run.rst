.. _run-source:

Run HiPACE++
============

Run a simulation
----------------

After compiling HiPACE++ (see above), from the HiPACE++ root directory, execute

.. code-block:: bash

   cd examples/linear_wake/
   ../../build/bin/hipace inputs # Run the simulation with fields and plasma and beam particles
   ./analysis.py # Plot results with yt, and save in img.png
