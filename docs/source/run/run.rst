.. _run-source:

Run Hipace++
============

Run a simulation
----------------

After compiling Hipace++ (see above), from the Hipace++ root directory, execute

.. code-block:: bash

   cd examples/linear_wake/
   ../../build/bin/hipace inputs # Run the simulation with fields and plasma and beam particles
   ./analysis.py # Plot results with yt, and save in img.png
