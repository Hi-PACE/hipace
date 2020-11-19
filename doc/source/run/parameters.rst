.. _parameters-source:

Input parameters
================

General parameters
------------------

* ``amr.n_cell`` (3 `integer`)
    The number of cells in x, y and z.

Plasma parameters
-----------------

* ``plasma.density`` (`float`) optional (default `0.`)
    The plasma density.

* ``plasma.ppc`` (2 `integer`) optional (default `0 0`)
    The number of plasma particles per cell in x and y.
    Since in a quasi-static code, there is only a 2D plasma slice evolving along the longitudinal
    coordinate, there is no need to specify a number of particles per cell in z.

* ``plasma.max_qsa_weighting_factor`` (`float`) optional (default `35.`)
    The maximum allowed weighting factor :math:`\gamma /(\psi+1)` before particles are considered
    as violating the quasi-static approximation and are removed from the simulation.

Beam parameters
---------------
Note: The following is just an example for how to import from doxygen.

.. doxygenclass:: PlasmaParticleContainer

.. doxygenclass:: BeamParticleContainer
