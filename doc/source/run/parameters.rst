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

   First, a fixed B-field error tolerance. This ensures the same level of convergence at each grid
   point. To do so, use e.g. the default settings of `hipace.predcorr_B_error_tolerance = 4e-2`,
   `hipace.predcorr_max_iterations = 30`, `hipace.predcorr_B_mixing_factor = 0.05`.
   This should almost always give reasonable results.

   Second, a fixed (low) number of iterations. This is usually much faster than the fixed B-field
   error, but can loose significant accuracy in special physical simulation settings. For most
   settings (e.g. a standard PWFA simulation the mild blowout regime at a reasonable resolution) it
   reproduces the same results as the fixed B-field error tolerance setting.
   A good setting for the fixed number of iterations is usually given by
   `hipace.predcorr_B_error_tolerance = -1.`, `hipace.predcorr_max_iterations = 1`,
   `hipace.predcorr_B_mixing_factor = 0.15`. The B-field error tolerance must be negative.

Plasma parameters
-----------------

For the plasma parameters, first the names of the plasmas need to be specified. Afterwards, the
plasma parameters for each plasma are specified via `plasma_name.plasma_property = ...`

* ``plasmas.names`` (`string`)
    The names of the plasmas, separated by a space.

* ``plasma_name.density`` (`float`) optional (default `0.`)
    The plasma density.

* ``plasma_name.ppc`` (2 `integer`) optional (default `0 0`)
    The number of plasma particles per cell in x and y.
    Since in a quasi-static code, there is only a 2D plasma slice evolving along the longitudinal
    coordinate, there is no need to specify a number of particles per cell in z.

* ``plasma_name.radius`` (`float`) optional (default `infinity`)
    Radius of the plasma. Set a value to run simulations in a plasma column.

* ``plasma_name.hollow_core_radius`` (`float`) optional (default `0.`)
    Inner radius of a hollow core plasma. The hollow core radius must be smaller than the plasma
    radius itself.

* ``plasma_name.parabolic_curvature`` (`float`) optional (default `0.`)
    Curvature of a parabolic plasma profile. The plasma density is set to
    :math:`\mathrm{plasma.density} * (1 + \mathrm{plasma.parabolic\_curvature}*r^2)`.

* ``plasma_name.max_qsa_weighting_factor`` (`float`) optional (default `35.`)
    The maximum allowed weighting factor :math:`\gamma /(\psi+1)` before particles are considered
    as violating the quasi-static approximation and are removed from the simulation.

* ``plasma_name.mass`` (`float`) optional (default `0.`)
    The mass of plasma particle in SI units. Use ``plasma_name.mass_Da`` for Dalton.
    Can also be set with ``plasma_name.element``. Must be > 0.

* ``plasma_name.mass_Da`` (`float`) optional (default `0.`)
    The mass of plasma particle in Dalton. Use ``plasma_name.mass`` for SI units.
    Can also be set with ``plasma_name.element``. Must be > 0.

* ``plasma_name.charge`` (`float`) optional (default `0.`)
    The charge of a plasma particle. Can also be set with plasma_name.element
    or if the plasma is Ionizable the default becomes :math:`+ q_e`.
    If the plasma is Ionizable, the charge gets multiplied by the current
    Ionization level.

* ``plasna_name.element`` (`string`) optional (default "")
    The Physical Element of the plasma. For "electron" and "positron" the charge
    and mass are set accordingly. For common Elements like "H", "He", "Li", ...
    the element is used to get the specific Ionization Energies of each state.

* ``plasma_name.can_ionize`` (`bool`) optional (default `0`)
    Whether this plasma can ionize. Can also be set by specifying
    ``plasma_name.initial_ion_level`` >= 0.

* ``plasma_name.initial_ion_level`` (`int`) optional (default `-1`)
    The initial Ionization state of the plasma. -1 for non-ionizable plasmas.
    0 for neutral, ionizable gasses and 1, 2, 3, ... for ionizable plasmas.

* ``plasma_name.ionization_product`` (`string`) optional (default "")
    The ``plasma_name`` of the plasma that contains the new electrons that are produced
    when this plasma gets ionized. Only needed if this plasma is ionizable.

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

* ``beam_name.num_particles`` (`int`)
    Number of constant weight particles to generate the beam.

* ``beam_name.duz_per_uz0_dzeta`` (`float`) optional (default `0.`)
    Relative correlated energy spread per :math:`\zeta`.
    Thereby, `duz_per_uz0_dzeta *` :math:`\zeta` `* uz_mean` is added to `uz` of the each particle.
    :math:`\zeta` is hereby the particle position relative to the mean
    longitudinal position of the beam.

* ``beam_name.do_symmetrize`` (`bool`) optional (default `0`)
    Symmetrizes the beam in the transverse phase space. For each particle with (`x`, `y`, `ux`,
    `uy`), three further particles are generated with (`-x`, `y`, `-ux`, `uy`), (`x`, `-y`, `ux`,
    `-uy`), and (`-x`, `-y`, `-ux`, `-uy`). The total number of particles will still be
    `beam_name.num_particles`, therefore this option requires that the beam particle number must be
    divisible by 4.

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

Diagnostic parameters
---------------------


* ``diagnostic.diag_type`` (`string`)
    Type of field output. Available options are `xyz`, `xz`, `yz`. `xyz` generates a 3D field
    output. Note that this can cause memory problems in particular on GPUs as the full 3D arrays
    need to be allocated. `xz` and `yz` generate 2D field outputs at the center of the y-axis and
    x-axis, respectively. In case of an even number of grid points, the value will be averaged
    between the two inner grid points.

* ``diagnostic.field_data`` (`string`) optional (default `all`)
    Names of the fields written to file, separated by a space. The field names need to be `all`,
    `none` or a subset of `ExmBy EypBx Ez Bx By Bz jx jy jz jx_beam jy_beam jz_beam rho Psi`.
    **Note:** The option `none` only suppressed the output of the field data. To suppress any
    output, please use `hipace.output_period = -1`.
