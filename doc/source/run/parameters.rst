.. _parameters-source:

Input parameters
================

General parameters
------------------

* ``amr.n_cell`` (3 `integer`)
    The number of cells in x, y and z.

* ``max_step`` (`integer`) optional (default `0.`)
    The maximum number of time steps.

* ``hipace.dt`` (`float`) optional (default `0.`)
    The time step to advance the particle beam.

* ``hipace.normalized_units`` (`bool`) optional (default `false`)
    Using normalized units in the simulation.

* ``hipace.verbose`` (`int`) optional (default `0`)
    The level of verbosity.

      * `verbose = 1`, prints only the time steps, which are computed.

      * `verbose = 2` additionally prints the number of iterations in the
        predictor-corrector loop, as well as the B-Field error at each slice.

      * `verbose = 4` also prints the number of particles, which violate the quasi-static
        approximation and were neglected at each slice.

* ``hipace.depos_order_xy`` (`int`) optional (default `2`)
    The transverse particle shape order. Currently, `0,1,2,3` are implemented.

* ``hipace.depos_order_z`` (`int`) optional (default `0`)
    The transverse particle shape order. Currently, only `0` is implemented.

Predictor-corrector loop parameters
-----------------------------------

* ``hipace.predcorr_B_error_tolerance`` (`float`) optional (default `4e-2`)
    The tolerance of the transverse B-field error. To enable a fixed number of iterations,
    `predcorr_B_error_tolerance` must be negative.

* ``hipace.m_predcorr_max_iterations`` (`int`) optional (default `5`)
    The maximum number of iterations in the predictor-corrector loop for single slice.

* ``hipace.predcorr_B_mixing_factor`` (`float`) optional (default `0.1`)
    The mixing factor between the currently calculated B-field and the B-field of the
    previous iteration (or initial guess, in case of the first iteration).
    A higher mixing factor leads to a faster convergence, but increases the chance of divergence.

.. note::
   In general, we recommend two different settings:

   First, a fixed B-field error tolerance. This ensures the same level of convergence at each grid point.
   To do so, use e.g. `hipace.predcorr_B_error_tolerance = 4e-2`, `hipace.m_predcorr_max_iterations = 30`,
   `hipace.predcorr_B_mixing_factor = 0.05`. This should almost always give reasonable results.

   Second, a fixed (low) number of iterations. This is usually much faster than the fixed B-field error,
   but can loose significant accuracy in special physical simulation settings. For most settings
   (e.g. a standard PWFA simulation the blowout regime) it reproduces the same results as the fixed
   B-field error tolerance setting.
   A good setting for the fixed number of iterations is usually given by
   `hipace.predcorr_B_error_tolerance = -1.`, `hipace.m_predcorr_max_iterations = 1`,
   `hipace.predcorr_B_mixing_factor = 0.15`. The B-field error tolerance *must* be negative.

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
