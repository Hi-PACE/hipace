.. _parameters-source:

Input parameters
================

General parameters
------------------

* ``amr.n_cell`` (3 `integer`)
    Number of cells in x, y and z.

* ``max_step`` (`integer`) optional (default `0`)
    Maximum number of time steps.

* ``hipace.dt`` (`float`) optional (default `0.`)
    Time step to advance the particle beam.

* ``hipace.normalized_units`` (`bool`) optional (default `0`)
    Using normalized units in the simulation.

* ``hipace.verbose`` (`int`) optional (default `0`)
    Level of verbosity.

      * `verbose = 1`, prints only the time steps, which are computed.

      * `verbose = 2` additionally prints the number of iterations in the
        predictor-corrector loop, as well as the B-Field error at each slice.

      * `verbose = 3` also prints the number of particles, which violate the quasi-static
        approximation and were neglected at each slice.

* ``hipace.depos_order_xy`` (`int`) optional (default `2`)
    Transverse particle shape order. Currently, `0,1,2,3` are implemented.

* ``hipace.depos_order_z`` (`int`) optional (default `0`)
    Transverse particle shape order. Currently, only `0` is implemented.

* ``hipace.output_period`` (`integer`) optional (default `-1`)
    | Output period. No output is given for `hipace.output_period = -1`.
    | **Warning:** `hipace.output_period = 0` will make the simulation crash.

* ``hipace.beam_injection_cr`` (`integer`) optional (default `1`)
    | Using a temporary coarsed grid for beam particle injection for a fixed particle-per-cell beam.
      For very high-resolution simulations, where the number of grid points (`nx*ny*nz`)
      exceeds the maximum `int (~2e9)`, it enables beam particle injection, which would
      fail otherwise. As an example, a simulation with `(2048*2048*2048)` grid points
      requires
    | `hipace.beam_injection_cr = 8`.

* ``hipace.do_beam_jx_jy_deposition`` (`bool`) optional (default `1`)
    Using the default, the beam deposits all currents `Jx`, `Jy`, `Jz`. Using
    `hipace.do_beam_jx_jy_deposition = 0` disables the transverse current deposition of the beams.

Predictor-corrector loop parameters
-----------------------------------

* ``hipace.predcorr_B_error_tolerance`` (`float`) optional (default `4e-2`)
    The tolerance of the transverse B-field error. To enable a fixed number of iterations,
    `predcorr_B_error_tolerance` must be negative.

* ``hipace.predcorr_max_iterations`` (`int`) optional (default `30`)
    The maximum number of iterations in the predictor-corrector loop for single slice.

* ``hipace.predcorr_B_mixing_factor`` (`float`) optional (default `0.05`)
    The mixing factor between the currently calculated B-field and the B-field of the
    previous iteration (or initial guess, in case of the first iteration).
    A higher mixing factor leads to a faster convergence, but increases the chance of divergence.

.. note::
   In general, we recommend two different settings:

   First, a fixed B-field error tolerance. This ensures the same level of convergence at each grid point.
   To do so, use e.g. the default settings of `hipace.predcorr_B_error_tolerance = 4e-2`,
   `hipace.predcorr_max_iterations = 30`, `hipace.predcorr_B_mixing_factor = 0.05`.
   This should almost always give reasonable results.

   Second, a fixed (low) number of iterations. This is usually much faster than the fixed B-field error,
   but can loose significant accuracy in special physical simulation settings. For most settings
   (e.g. a standard PWFA simulation the blowout regime) it reproduces the same results as the fixed
   B-field error tolerance setting.
   A good setting for the fixed number of iterations is usually given by
   `hipace.predcorr_B_error_tolerance = -1.`, `hipace.predcorr_max_iterations = 1`,
   `hipace.predcorr_B_mixing_factor = 0.15`. The B-field error tolerance must be negative.

Plasma parameters
-----------------

* ``plasma.density`` (`float`) optional (default `0.`)
    The plasma density.

* ``plasma.ppc`` (2 `integer`) optional (default `0 0`)
    The number of plasma particles per cell in x and y.
    Since in a quasi-static code, there is only a 2D plasma slice evolving along the longitudinal
    coordinate, there is no need to specify a number of particles per cell in z.

* ``plasma.radius`` (`float`) optional (default `infinity`)
    Radius of the plasma. Set a value to run simulations in a plasma column.

* ``plasma.channel_radius`` (`float`) optional (default `0.`)
    Channel radius of a parabolic plasma profile. The plasma density is set to
    :math:`\mathrm{plasma.density} * (1 + r^2/\mathrm{plasma.channel\_radius}^2)`.

* ``plasma.max_qsa_weighting_factor`` (`float`) optional (default `35.`)
    The maximum allowed weighting factor :math:`\gamma /(\psi+1)` before particles are considered
    as violating the quasi-static approximation and are removed from the simulation.

Beam parameters
---------------

For the beam parameters, first the names of the beams need to be specified. Afterwards, the beam
parameters for each beam are specified via `beam_name.beam_property = ...`

* ``beams.names`` (`string`)
    The names of the particle beams, separated by a space.

* ``beam_name.injection_type`` (`string`)
    The injection type for the particle beam. Currently available are `fixed_ppc`, `fixed_weight`,
    and `from_file`. `fixed_ppc` generates a beam with a fixed number of particles per cell and
    varying weights. `fixed_weight` generates a beam with a fixed number of particles with a
    constant weight. `from_file` reads a beam from openPMD files.

**fixed_weight**

* ``beam_name.duz_per_uz0_dzeta`` (`float`) optional (default `0.`)
    Relative correlated energy spread per :math:`\zeta`. The correlated energy spread is
    calculated with respect to the mean longitudinal position of the beam, so the mean position in z
    has the mean energy.

**from_file**

* ``beam_name.input_file`` (`string`)
    Name of the input file. **Note:** Reading in files with digits in their names (e.g.
    `openpmd_002135.h5`) can be problematic, it is advised to read them via `openpmd_%T.h5` and then
    specify the iteration via `beam_name.iteration = 2135`.

* ``beam_name.iteration`` (`integer`) optional (default `0`)
    Iteration of the openPMD file to be read in. If the openPMD file contains multiple iterations,
    or multiple openPMD files are read in, the iteration can be specified. **Note:** The physical
    time of the simulation is set to the time of the given iteration (if available).

* ``beam_name.openPMD_species_name`` (`string`) optional
    Name of the beam to be read in. If an openPMD file contains multiple beams, the name of the beam
    needs to be specified.
