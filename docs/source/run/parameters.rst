.. _parameters-source:

Input parameters
================

Parser
------

In HiPACE++ all input parameters are obtained through ``amrex::Parser``, making it possible to
specify input parameters with expressions and not just numbers. User constants can be defined
in the input script with ``my_constants``.

.. code-block:: bash

    my_constants.ne = 1.25e24
    my_constants.kp_inv = "clight / sqrt(ne * q_e^2  / (epsilon0 * m_e))"
    beam.radius = "kp_inv / 2"

Thereby, the following constants are predefined:

============ =================== ================= ====================
**variable** **name**            **SI value**      **normalized value**
q_e          elementary charge   1.602176634e-19   1
m_e          electron mass       9.1093837015e-31  1
m_p          proton mass         1.67262192369e-27 1836.15267343
epsilon0     vacuum permittivity 8.8541878128e-12  1
mu0          vacuum permeability 1.25663706212e-06 1
clight       speed of light      299'792'458.      1
============ =================== ================= ====================

For a list of supported functions see the
`AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parser>`__.
Sometimes it is necessary to use double-quotes around expressions, especially when providing them
as command line parameters. Multi-line expressions are allowed if surrounded by double-quotes.

General parameters
------------------

* ``amr.n_cell`` (3 `integer`)
    Number of cells in x, y and z.
    With the explicit solver (default), the number of cells in the x and y directions must be either :math:`2^n-1` (common values are 511, 1023, 2047, best configuration for performance) or :math:`2^n` where :math:`n` is an integer. Some other values might work, like :math:`3 \times 2^n-1`, but use at your own risk.

* ``amr.max_level`` (`integer`) optional (default `0`)
    Maximum level of mesh refinement. Currently, mesh refinement is supported up to level
    `2`. Note, that the mesh refinement algorithm is still in active development and should be used with care.

* ``geometry.patch_lo`` (3 `float`)
    Lower end of the simulation box in x, y and z.

* ``geometry.patch_hi`` (3 `float`)
    Higher end of the simulation box in x, y and z.

* ``geometry.is_periodic`` (3 `bool`)
    Whether the boundary conditions for particles in x, y and z is periodic. Note that particles in z are always removed. This setting will most likely be changed in the near future.

* ``mr_lev1.n_cell`` (2 `integer`)
    Number of cells in x and y for level 1.
    The number of cells in the zeta direction is calculated from ``patch_lo`` and ``patch_hi``.

* ``mr_lev1.patch_lo`` (3 `float`)
    Lower end of the refined grid in x, y and z.

* ``mr_lev1.patch_hi`` (3 `float`)
    Upper end of the refined grid in x, y and z.

* ``mr_lev2.n_cell`` (2 `integer`)
    Number of cells in x and y for level 2.
    The number of cells in the zeta direction is calculated from ``patch_lo`` and ``patch_hi``.

* ``mr_lev2.patch_lo`` (3 `float`)
    Lower end of the refined grid in x, y and z.

* ``mr_lev2.patch_hi`` (3 `float`)
    Upper end of the refined grid in x, y and z.

* ``max_step`` (`integer`) optional (default `0`)
    Maximum number of time steps. `0` means that the 0th time step will be calculated, which are the
    fields of the initial beams.

* ``random_seed`` (`integer`) optional (default `1`)
    Passes a seed to the AMReX random number generator. This allows for reproducibility of random events such as randomly generated beams, ionization, and collisions.
    Note that on GPU, since the order of operations is not ensured, the providing of a seed does not guarantee reproducibility to the level of machine precision.

* ``hipace.max_time`` (`float`) optional (default `infinity`)
    Maximum physical time of the simulation. The ``dt`` of the last time step may be reduced so that ``t + dt = max_time``, both for the adaptive and a fixed time step.

* ``hipace.dt`` (`float` or `string`) optional (default `0.`)
    Time step to advance the particle beam. For adaptive time step, use ``"adaptive"``.

* ``hipace.dt_max`` (`float`) optional (default `inf`)
    Only used if ``hipace.dt = adaptive``. Upper bound of the adaptive time step: if the computed adaptive time step is is larger than ``dt_max``, then ``dt_max`` is used instead.
    Useful when the plasma profile starts with a very low density (e.g. in the presence of a realistic density ramp), to avoid unreasonably large time steps.

* ``hipace.nt_per_betatron`` (`Real`) optional (default `20.`)
    Only used when using adaptive time step (see ``hipace.dt`` above).
    Number of time steps per betatron period (of the full blowout regime).
    The time step is given by :math:`\omega_{\beta}\Delta t = 2 \pi/N`
    (:math:`N` is ``nt_per_betatron``) where :math:`\omega_{\beta}=\omega_p/\sqrt{2\gamma}` with
    :math:`\omega_p` the plasma angular frequency and :math:`\gamma` is an average of Lorentz
    factors of the slowest particles in all beams.

* ``hipace.adaptive_predict_step`` (`bool`) optional (default `1`)
    Only used when using adaptive time step (see ``hipace.dt`` above).
    If true, the current Lorentz factor and accelerating field on the beams are used to predict the (adaptive) ``dt`` of the next time steps.
    This prediction is used to better estimate the betatron frequency at the beginning of the next step performed by the current rank.
    It improves accuracy for parallel simulations (with significant deceleration and/or z-dependent plasma profile).
    Note: should be on by default once good defaults are determined.

* ``hipace.adaptive_control_phase_advance`` (`bool`) optional (default `1`)
    Only used when using adaptive time step (see ``hipace.dt`` above).
    If true, a test on the phase advance sets the time step so it matches the phase advance expected for a uniform plasma (to a certain tolerance).
    This should improve the accuracy in the presence of density gradients.
    Note: should be on by default once good defaults are determined.

* ``hipace.adaptive_phase_tolerance`` (`Real`) optional (default `4.e-4`)
    Only used when using adaptive time step (see ``hipace.dt`` above) and ``adaptive_control_phase_advance``.
    Tolerance for the controlled phase advance described above (lower is more accurate, but should result in more time steps).

* ``hipace.adaptive_phase_substeps`` (`int`) optional (default `2000`)
    Only used when using adaptive time step (see ``hipace.dt`` above) and ``adaptive_control_phase_advance``.
    Number of sub-steps in the controlled phase advance described above (higher is more accurate, but should be slower).

* ``hipace.adaptive_threshold_uz`` (`Real`) optional (default `2.`)
    Only used when using adaptive time step (see ``hipace.dt`` above).
    Threshold beam momentum, below which the time step is not decreased (to avoid arbitrarily small time steps).

* ``hipace.normalized_units`` (`bool`) optional (default `0`)
    Using normalized units in the simulation.

* ``hipace.verbose`` (`int`) optional (default `0`)
    Level of verbosity.

      * ``hipace.verbose = 1``, prints only the time steps, which are computed.

      * ``hipace.verbose = 2`` additionally prints the number of iterations in the
        predictor-corrector loop, as well as the B-Field error at each slice.

      * ``hipace.verbose = 3`` also prints the number of particles, which violate the quasi-static
        approximation and were neglected at each slice. It prints the number of ionized particles,
        if ionization occurred. It also adds additional information if beams
        are read in from file.

* ``hipace.do_device_synchronize`` (`int`) optional (default `0`)
    Level of synchronization on GPU.

      * ``hipace.do_device_synchronize = 0``, synchronization happens only when necessary.

      * ``hipace.do_device_synchronize = 1``, synchronizes most functions (all that are profiled
        via ``HIPACE_PROFILE``)

      * ``hipace.do_device_synchronize = 2`` additionally synchronizes low-level functions (all that
        are profiled via ``HIPACE_DETAIL_PROFILE``)

* ``hipace.depos_order_xy`` (`int`) optional (default `2`)
    Transverse particle shape order. Currently, `0,1,2,3` are implemented.

* ``hipace.depos_order_z`` (`int`) optional (default `0`)
    Longitudinal particle shape order. Currently, only `0` is implemented.

* ``hipace.depos_derivative_type`` (`int`) optional (default `2`)
    Type of derivative used in explicit deposition. `0`: analytic, `1`: nodal, `2`: centered

* ``hipace.outer_depos_loop`` (`bool`) optional (default `0`)
    If the loop over depos_order is included in the loop over particles.

* ``hipace.beam_injection_cr`` (`integer`) optional (default `1`)
    Using a temporary coarsed grid for beam particle injection for a fixed particle-per-cell beam.
    For very high-resolution simulations, where the number of grid points (`nx*ny*nz`)
    exceeds the maximum `int (~2e9)`, it enables beam particle injection, which would
    fail otherwise. As an example, a simulation with `2048 x 2048 x 2048` grid points
    requires ``hipace.beam_injection_cr = 8``.

* ``hipace.do_beam_jx_jy_deposition`` (`bool`) optional (default `1`)
    Using the default, the beam deposits all currents ``Jx``, ``Jy``, ``Jz``. Using
    ``hipace.do_beam_jx_jy_deposition = 0`` disables the transverse current deposition of the beams.

* ``hipace.boxes_in_z`` (`int`) optional (default `1`)
    Number of boxes along the z-axis. In serial runs, the arrays for 3D IO can easily exceed the
    memory of a GPU. Using multiple boxes reduces the memory requirements by the same factor.
    This option is only available in serial runs, in parallel runs, please use more GPU to achieve
    the same effect.

* ``hipace.openpmd_backend`` (`string`) optional (default `h5`)
    OpenPMD backend. This can either be ``h5``, ``bp``, or ``json``. The default is chosen by what is
    available. If both Adios2 and HDF5 are available, ``h5`` is used. Note that ``json`` is extremely
    slow and is not recommended for production runs.

* ``hipace.file_prefix`` (`string`) optional (default `diags/hdf5/`)
    Path of the output.

* ``hipace.do_tiling`` (`bool`) optional (default `true`)
    Whether to use tiling, when running on CPU.
    Currently, this option only affects plasma operations (gather, push and deposition).
    The tile size can be set with ``plasmas.sort_bin_size``.

* ``hipace.do_beam_jz_minus_rho`` (`bool`) optional (default `0`)
    Whether the beam contribution to :math:`j_z-c\rho` is calculated and used when solving for Psi (used to caculate the transverse fields Ex-By and Ey+Bx).
    if 0, this term is assumed to be 0 (a good approximation for an ultra-relativistic beam in the z direction with small transverse momentum).

* ``hipace.deposit_rho`` (`bool`) optional (default `0`)
    If the charge density ``rho`` of the plasma should be deposited so that it is available as a diagnostic.
    Otherwise only ``rhomjz`` equal to :math:`\rho-j_z/c` will be available.
    If ``rho`` is explicitly mentioned in ``diagnostic.field_data``, then the default will become `1`.

* ``hipace.salame_n_iter`` (`int`) optional (default `3`)
    Number of iterations the SALAME algorithm should do when it is used.

* ``hipace.salame_do_advance`` (`bool`) optional (default `1`)
    Whether the SALAME algorithm should calculate the SALAME-beam-only Ez field
    by advancing plasma (if `1`) particles or by approximating it using the chi field (if `0`).

* ``hipace.salame_Ez_target(zeta,zeta_initial,Ez_initial)`` (`string`) optional (default `Ez_initial`)
    Parser function to specify the target Ez field at the witness beam for SALAME.
    ``zeta``: position of the Ez field to set.
    ``zeta_initial``: position where the SALAME algorithm first started.
    ``Ez_initial``: field value at `zeta_initial`.
    For `zeta` equal to `zeta_initial`, the function should return `Ez_initial`.
    The default value of this function corresponds to a flat Ez field at the position of the SALAME beam.
    Note: `zeta` is always less than or equal to `zeta_initial` and `Ez_initial` is typically below zero for electron beams.

Field solver parameters
-----------------------

Two different field solvers are available to calculate the transverse magnetic fields `Bx`
and `By`: an explicit solver (based on analytic integration) and a predictor-corrector loop (based on an FFT solver).
In the explicit solver, the longitudinal derivative of the transverse currents is calculated explicitly, which
results in a shielded Poisson equation, solved with either the internal HiPACE++ multigrid solver or the AMReX multigrid solver.
The default is to use the explicit solver. **We strongly recommend to use the explicit solver**, because we found it to be more robust, faster to converge, and easier to use.


* ``hipace.bxby_solver`` (`string`) optional (default `explicit`)
    Which solver to use.
    Possible values: ``explicit`` and ``predictor-corrector``.

* ``hipace.use_small_dst`` (`bool`) optional (default `0` or `1`)
    Whether to use a large R2C or a small C2R fft in the dst of the Poisson solver.
    The small dst is quicker for simulations with :math:`\geq 511` transverse grid points.
    The default is set accordingly.

* ``fields.extended_solve`` (`bool`) optional (default `0`)
    Extends the area of the FFT Poisson solver to the ghost cells. This can reduce artifacts
    originating from the boundary for long simulations.

* ``fields.open_boundary`` (`bool`) optional (default `0`)
    Uses a Taylor approximation of the Greens function to solve the Poisson equations with
    open boundary conditions. It's recommended to use this together with
    ``fields.extended_solve = true`` and ``geometry.is_periodic = false false false``.
    Only available with the predictor-corrector solver.

Explicit solver parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``hipace.use_amrex_mlmg`` (`bool`) optional (default `0`)
    Whether to use the AMReX multigrid solver. Note that this requires the compile-time option ``AMReX_LINEAR_SOLVERS`` to be true. Generally not recommended since it is significantly slower than the default HiPACE++ multigrid solver.

* ``hipace.MG_tolerance_rel`` (`float`) optional (default `1e-4`)
    Relative error tolerance of the multigrid solvers.

* ``hipace.MG_tolerance_abs`` (`float`) optional (default `0.`)
    Absolute error tolerance of the multigrid solvers.

* ``hipace.MG_verbose`` (`int`) optional (default `0`)
    Level of verbosity of the the multigrid solvers.

Predictor-corrector loop parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``hipace.predcorr_B_error_tolerance`` (`float`) optional (default `4e-2`)
    The tolerance of the transverse B-field error. Set to a negative value to use a fixed number of iterations.

* ``hipace.predcorr_max_iterations`` (`int`) optional (default `30`)
    The maximum number of iterations in the predictor-corrector loop for single slice.

* ``hipace.predcorr_B_mixing_factor`` (`float`) optional (default `0.05`)
    The mixing factor between the currently calculated B-field and the B-field of the
    previous iteration (or initial guess, in case of the first iteration).
    A higher mixing factor leads to a faster convergence, but increases the chance of divergence.

.. note::
   In general, we recommend two different settings:

   First, a fixed B-field error tolerance. This ensures the same level of convergence at each grid
   point. To do so, use e.g. the default settings of ``hipace.predcorr_B_error_tolerance = 4e-2``,
   ``hipace.predcorr_max_iterations = 30``, ``hipace.predcorr_B_mixing_factor = 0.05``.
   This should almost always give reasonable results.

   Second, a fixed (low) number of iterations. This is usually much faster than the fixed B-field
   error, but can loose significant accuracy in special physical simulation settings. For most
   settings (e.g. a standard PWFA simulation the blowout regime at a reasonable resolution) it
   reproduces the same results as the fixed B-field error tolerance setting. It works very well at
   high longitudinal resolution.
   A good setting for the fixed number of iterations is usually given by
   ``hipace.predcorr_B_error_tolerance = -1.``, ``hipace.predcorr_max_iterations = 1``,
   ``hipace.predcorr_B_mixing_factor = 0.15``. The B-field error tolerance must be negative.


Plasma parameters
-----------------

The name of all plasma species must be specified with `plasmas.names = ...`.
Then, properties can be set per plasma species with ``<plasma name>.<plasma property> = ...``,
or sometimes for all plasma species at the same time with ``plasmas.<plasma property> = ...``.
When both are specified, the per-species value is used.

* ``plasmas.names`` (`string`) optional (default `no_plasma`)
    The names of the plasmas, separated by a space.
    To run without plasma, choose the name ``no_plasma``.

* ``<plasma name> or plasmas.density(x,y,z)`` (`float`) optional (default `0.`)
    The plasma density as function of `x`, `y` and `z`. `x` and `y` coordinates are taken from
    the simulation box and :math:`z = time \cdot c`. The density gets recalculated at the beginning
    of every timestep. If specified as a command line parameter, quotation marks must be added:
    ``"<plasma name>.density(x,y,z)" = "1."``.

* ``<plasma name> or plasmas.min_density`` (`float`) optional (default `0`)
    Particles with a density less than or equal to the minimal density won't be injected.
    Useful for parsed functions to avoid redundant plasma particles with close to 0 weight.

* ``<plasma name>.density_table_file`` (`string`) optional (default "")
    Alternative to ``<plasma name>.density(x,y,z)``. Specify the name of a text file containing
    multiple densities for different positions. File syntax: ``<position> <density function>`` for
    every line. If a line doesn't start with a position it is ignored (comments can be made
    with `#`). `<density function>` is evaluated like ``<plasma name>.density(x,y,z)``. The simulation
    position :math:`time \cdot c` is rounded up to the nearest `<position>` in the file to get it's
    `<density function>` which is used for that time step.

* ``<plasma name> or plasmas.ppc`` (2 `integer`) optional (default `0 0`)
    The number of plasma particles per cell in x and y.
    Since in a quasi-static code, there is only a 2D plasma slice evolving along the longitudinal
    coordinate, there is no need to specify a number of particles per cell in z.

* ``<plasma name> or plasmas.radius`` (`float`) optional (default `infinity`)
    Radius of the plasma. Set a value to run simulations in a plasma column.

* ``<plasma name> or plasmas.hollow_core_radius`` (`float`) optional (default `0.`)
    Inner radius of a hollow core plasma. The hollow core radius must be smaller than the plasma
    radius itself.

* ``<plasma name> or plasmas.max_qsa_weighting_factor`` (`float`) optional (default `35.`)
    The maximum allowed weighting factor :math:`\gamma /(\psi+1)` before particles are considered
    as violating the quasi-static approximation and are removed from the simulation.

* ``<plasma name>.mass`` (`float`) optional (default `0.`)
    The mass of plasma particle in SI units. Use ``plasma_name.mass_Da`` for Dalton.
    Can also be set with ``<plasma name>.element``. Must be `>0`.

* ``<plasma name>.mass_Da`` (`float`) optional (default `0.`)
    The mass of plasma particle in Dalton. Use ``<plasma name>.mass`` for SI units.
    Can also be set with ``<plasma name>.element``. Must be `>0`.

* ``<plasma name>.charge`` (`float`) optional (default `0.`)
    The charge of a plasma particle. Can also be set with ``<plasma name>.element``.
    The charge gets multiplied by the current ionization level.

* ``<plasma name>.element`` (`string`) optional (default "")
    The physical element of the plasma. Sets charge, mass and, if available,
    the specific ionization energy of each state.
    Options are: ``electron``, ``positron``, ``H``, ``D``, ``T``, ``He``, ``Li``, ``Be``, ``B``, ….

* ``<plasma name>.can_ionize`` (`bool`) optional (default `0`)
    Whether this plasma can ionize. Can also be set to 1 by specifying ``<plasma name>.ionization_product``.

* ``<plasma name>.initial_ion_level`` (`int`) optional (default `-1`)
    The initial ionization state of the plasma. `0` for neutral gasses.
    If set, the plasma charge gets multiplied by this number.

* ``<plasma name>.ionization_product`` (`string`) optional (default "")
    Name of the plasma species that contains the new electrons that are produced
    when this plasma gets ionized. Only needed if this plasma is ionizable.

* ``<plasma name> or plasmas.neutralize_background`` (`bool`) optional (default `1`)
    Whether to add a neutralizing background of immobile particles of opposite charge.

* ``plasmas.sort_bin_size`` (`int`) optional (default `32`)
    Tile size for plasma current deposition, when running on CPU.
    When tiling is activated (``hipace.do_tiling = 1``), the current deposition is done in temporary
    arrays of size ``sort_bin_size`` (+ guard cells) that are atomic-added to the main current
    arrays.

* ``<plasma name>.temperature_in_ev`` (`float`) optional (default `0`)
    | Initializes the plasma particles with a given temperature :math:`k_B T` in eV. Using a temperature, the plasma particle momentum is normally distributed with a variance of :math:`k_B T /(M c^2)` in each dimension, with :math:`M` the particle mass, :math:`k_B` the Boltzmann constant, and :math:`T` the isotropic temperature in Kelvin.
    | Note: Using a temperature can affect the performance since the plasma particles loose their order and thus their favorable memory access pattern. The performance can be mostly recovered by reordering the plasma particles (see ``<plasma name> or plasmas.reorder_period``).
      Furthermore, the noise of the temperature can seed the hosing instability. The amplitude of the seeding is unphysical, because the number of macro-particles is typically orders of magnitude below the number of actual plasma electrons.
      Since it is often unfeasible to use a sufficient amount of plasma macro-particles per cell to suppress this numerical seed, the plasma can be symmetrized to prevent the onset of the hosing instability (see ``<plasma name> or plasmas.do_symmetrize``).

* ``<plasma name> or plasmas.do_symmetrize`` (`bool`) optional (default `0`)
    Symmetrizes the plasma in the transverse phase space. For each particle with (`x`, `y`, `ux`,
    `uy`), three additional particles are generated with (`-x`, `y`, `-ux`, `uy`), (`x`, `-y`, `ux`,
    `-uy`), and (`-x`, `-y`, `-ux`, `-uy`).
    The total number of plasma particles is multiplied by 4. This option is helpful to prevent a numerical seeding of the hosing instability for a plasma with a temperature.

* ``<plasma name> or plasmas.reorder_period`` (`int`) optional (default `0`)
    Reorder particles periodically to speed-up current deposition on GPU for a high-temperature plasma.
    A good starting point is a period of 4 to reorder plasma particles on every fourth zeta-slice.
    To disable reordering set this to 0.

* ``<plasma name> or plasmas.reorder_idx_type`` (2 `int`) optional (default `0 0` or `1 1`)
    Change if plasma particles are binned to cells (0), nodes (1) or both (2)
    for both x and y direction as part of the reordering.
    The ideal index type depends on the particle shape factor used for deposition.
    For shape factors 1 and 3, 2^2 and 4^2 cells are deposited per particle respectively,
    resulting in node centered reordering giving better performance.
    For shape factors 0 and 2, 1^2 and 3^2 cells are deposited such that cell centered reordering is better.
    The default is chosen accordingly.
    If ``hipace.depos_derivative_type = 1``, the explicit deposition deposits an additional cell in each direction,
    making the opposite index type ideal. Since the normal deposition still requires the original index type,
    the compromise option ``2 2`` can be chosen. This will however require more memory in the binning process.

Binary collisions for plasma species
------------------------------------

WARNING: this module is in development.

HiPACE++ proposes an implementation of [Perez et al., Phys. Plasmas 19, 083104 (2012)], inherited from WarpX, between plasma species.

* ``plasmas.background_density_SI`` (`float`) optional
    Background plasma density in SI units. Only used for collisions in normalized units. Since the collision rate depends on the plasma density itself, it cannot be determined in normalized units without knowing the actual plasma background density.
    Hence, it must be provided using this input parameter.

* ``plasmas.collisions`` (list of `strings`) optional
    List of names of types binary Coulomb collisions.
    Each will represent collisions between 2 plasma species (potentially the same).

* ``<collision name>.species`` (two `strings`) optional
    The name of the two plasma species for which collisions should be included.

* ``<collision name>.CoulombLog`` (`float`) optional (default `-1.`)
    Coulomb logarithm used for this collision.
    If not specified, the Coulomb logarithm is determined from the temperature in each cell.

Beam parameters
---------------

For the beam parameters, first the names of the beams need to be specified. Afterwards, the beam
parameters for each beam are specified via ``<beam name>.<beam property> = ...``

* ``beams.names`` (`string`) optional (default `no_beam`)
    The names of the particle beams, separated by a space.
    To run without beams, choose the name ``no_beam``.

General beam parameters
^^^^^^^^^^^^^^^^^^^^^^^
The general beam parameters are applicable to all particle beam types. More specialized beam parameters,
which are valid only for certain beam types, are introduced further below under
"Option: ``<injection_type>``".


* ``<beam name>.injection_type`` (`string`)
    The injection type for the particle beam. Currently available are ``fixed_ppc``, ``fixed_weight``,
    and ``from_file``. ``fixed_ppc`` generates a beam with a fixed number of particles per cell and
    varying weights. It can be either a Gaussian or a flattop beam. ``fixed_weight`` generates a
    Gaussian beam with a fixed number of particles with a constant weight.
    ``from_file`` reads a beam from openPMD files.

* ``<beam name>.position_mean`` (3 `float`)
    The mean position of the beam in ``x, y, z``, separated by a space. For fixed_weight beams the
    x and y directions can be functions of ``z``. To generate a tilted beam use
    ``<beam name>.position_mean = "x_center+(z-z_ center)*dx_per_dzeta" "y_center+(z-z_ center)*dy_per_dzeta" "z_center"``.

* ``<beam name>.position_std`` (3 `float`)
    The rms size of the of the beam in `x, y, z`, separated by a space.

* ``<beam name>.zmin`` (`float`) (default `-infinity`)
    Minimum in `z` at which particles are injected.

* ``<beam name>.zmax`` (`float`) (default `infinity`)
    Maximum in `z` at which particles are injected.

* ``<beam name>.element`` (`string`) optional (default `electron`)
    The Physical Element of the plasma. Sets charge, mass and, if available,
    the specific Ionization Energy of each state.
    Currently available options are: ``electron``, ``positron``, and ``proton``.

* ``<beam name>.mass`` (`float`) optional (default `m_e`)
    The mass of beam particles. Can also be set with ``<beam name>.element``. Must be `>0`.

* ``<beam name>.charge`` (`float`) optional (default `-q_e`)
    The charge of a beam particle. Can also be set with ``<beam name>.element``.

* ``<beam name>.profile`` (`string`)
    Beam profile.
    When ``<beam name>.injection_type == fixed_ppc``, possible options are ``flattop``
    (flat-top radially and longitudinally), ``gaussian`` (Gaussian in all directions),
    or ``parsed`` (arbitrary analytic function provided by the user).
    When ``parsed``, ``<beam name>.density(x,y,z)`` must be specified.
    When ``<beam name>.injection_type == fixed_weight``, possible options are ``can``
    (uniform longitudinally, Gaussian transversally) and ``gaussian`` (Gaussian in all directions).

* ``<beam name>.n_subcycles`` (`int`) optional (default `10`)
    Number of sub-cycles performed in the beam particle pusher. The particles will be pushed
    ``n_subcycles`` times with a time step of `dt/n_subcycles`. This can be used to improve accuracy
    in highly non-linear focusing fields.

* ``<beam name>.do_salame`` (`bool`) optional (default `0`)
    Whether to use the SALAME algorithm [S. Diederichs et al., Phys. Rev. Accel. Beams 23, 121301 (2020)] to automatically flatten the accelerating field in the first time step. If turned on, the per-slice
    beam weight in the first time-step is adjusted such that the Ez field will be uniform in the beam.
    This will ignore the contributions to jx, jy and rho from the beam in the first time-step.
    It is recommended to use this option with a fixed weight can beam.
    If a gaussian beam profile is used, then the zmin and zmax parameters should be used.

* ``hipace.external_E_uniform`` (3 `float`) optional (default `0. 0. 0.`)
    Uniform external electric field applied to beam particles.
    The components represent Ex-c*By, Ey+c*Bx and Ez respectively.

* ``hipace.external_B_uniform`` (3 `float`) optional (default `0. 0. 0.`)
    Uniform external magnetic field applied to beam particles.
    The components represent Bx, By and Bz, respectively.

* ``hipace.external_E_slope`` (3 `float`) optional (default `0. 0. 0.`)
    Slope of a linear external electric field applied to beam particles.
    The components represent d(Ex-c*By)/dx, d(Ey+c*Bx)/dy and d(Ez)/dz respectively.
    For the last component, z actually represents the zeta coordinate zeta = z - c*t.

* ``hipace.external_B_slope`` (3 `float`) optional (default `0. 0. 0.`)
    Slope of a linear external electric field applied to beam particles.
    The components represent d(Bx)/dy, d(By)/dx and d(Bz)/dz respectively.
    Note the order of derivatives for the transverse components!
    For the last component, z actually represents the zeta coordinate zeta = z - c*t.
    For instance, ``hipace.external_B_slope = -1. 1. 0.`` creates an axisymmetric focusing lens of strength 1 T/m.

Option: ``fixed_weight``
^^^^^^^^^^^^^^^^^^^^^^^^

* ``<beam name>.num_particles`` (`int`)
    Number of constant weight particles to generate the beam.

* ``<beam name>.total_charge`` (`float`)
    Total charge of the beam. Note: Either ``total_charge`` or ``density`` must be specified.
    The absolute value of this parameter is used when initializing the beam.
    Note that ``<beam name>.zmin`` and ``<beam name>.zmax`` can reduce the total charge.

* ``<beam name>.density`` (`float`)
    Peak density of the beam. Note: Either ``total_charge`` or ``density`` must be specified.
    The absolute value of this parameter is used when initializing the beam.

* ``<beam name>.duz_per_uz0_dzeta`` (`float`) optional (default `0.`)
    Relative correlated energy spread per :math:`\zeta`.
    Thereby, `duz_per_uz0_dzeta *` :math:`\zeta` `* uz_mean` is added to `uz` of the each particle.
    :math:`\zeta` is hereby the particle position relative to the mean
    longitudinal position of the beam.

* ``<beam name>.do_symmetrize`` (`bool`) optional (default `0`)
    Symmetrizes the beam in the transverse phase space. For each particle with (`x`, `y`, `ux`,
    `uy`), three further particles are generated with (`-x`, `y`, `-ux`, `uy`), (`x`, `-y`, `ux`,
    `-uy`), and (`-x`, `-y`, `-ux`, `-uy`). The total number of particles will still be
    ``beam_name.num_particles``, therefore this option requires that the beam particle number must be
    divisible by 4.

* ``<beam name>.do_z_push`` (`bool`) optional (default `1`)
    Whether the beam particles are pushed along the z-axis. The momentum is still fully updated.
    Note: using ``do_z_push = 0`` results in unphysical behavior.

* ``<beam name>.z_foc`` (`float`) optional (default `0.`)
    Distance at which the beam will be focused, calculated from the position at which the beam is initialized.
    The beam is assumed to propagate ballistically in-between.

Option: ``fixed_ppc``
^^^^^^^^^^^^^^^^^^^^^

* ``<beam name>.ppc`` (3 `int`) (default `1 1 1`)
    Number of particles per cell in `x`-, `y`-, and `z`-direction to generate the beam.

* ``<beam name>.radius`` (`float`)
    Maximum radius ``<beam name>.radius`` :math:`= \sqrt{x^2 + y^2}` within that particles are
    injected.

* ``<beam name>.density`` (`float`)
    Peak density of the beam.
    The absolute value of this parameter is used when initializing the beam.

* ``<beam name>.density(x,y,z)`` (`float`)
    The density profile of the beam, as a function of spatial dimensions `x`, `y` and `z`.
    This function uses the parser, see above.
    Only used when ``<beam name>.profile == parsed``.

* ``<beam name>.min_density`` (`float`) optional (default `0`)
    Minimum density. Particles with a lower density are not injected.
    The absolute value of this parameter is used when initializing the beam.

* ``<beam name>.random_ppc`` (3 `bool`) optional (default `0 0 0`)
    Whether the position in `(x y z)` of the particles is randomized within the cell.

Option: ``from_file``
^^^^^^^^^^^^^^^^^^^^^

* ``<beam name> or beams.input_file`` (`string`)
    Name of the input file. **Note:** Reading in files with digits in their names (e.g.
    ``openpmd_002135.h5``) can be problematic, it is advised to read them via ``openpmd_%T.h5`` and then
    specify the iteration via ``beam_name.iteration = 2135``.

* ``<beam name> or beams.iteration`` (`integer`) optional (default `0`)
    Iteration of the openPMD file to be read in. If the openPMD file contains multiple iterations,
    or multiple openPMD files are read in, the iteration can be specified. **Note:** The physical
    time of the simulation is set to the time of the given iteration (if available).

* ``<beam name>.openPMD_species_name`` (`string`) optional (default `<beam name>`)
    Name of the beam to be read in. If an openPMD file contains multiple beams, the name of the beam
    needs to be specified.

Laser parameters
----------------

The laser profile is defined by :math:`a(x,y,z) = a_0 * \mathrm{exp}[-(x^2/w0_x^2 + y^2/w0_y^2 + z^2/L0^2)]`.
The model implemented is the one from [C. Benedetti et al. Plasma Phys. Control. Fusion 60.1: 014002 (2017)].
Unlike for ``beams`` and ``plasmas``, all the laser pulses are currently stored on the same array,
which you can find in the output openPMD file as `laser_real` (for the real part of the envelope) and `laser_imag` for its imaginary part.
Parameters starting with ``lasers.`` apply to all laser pulses, parameters starting with ``<laser name>`` apply to a single laser pulse.

* ``lasers.names`` (list of `string`) optional (default `no_laser`)
    The names of the laser pulses, separated by a space.
    To run without a laser, choose the name ``no_laser``.

* ``lasers.lambda0`` (`float`)
    Wavelength of the laser pulses. Currently, all pulses must have the same wavelength.

* ``lasers.use_phase`` (`bool`) optional (default `true`)
    Whether the phase terms (:math:`\theta` in Eq. (6) of [C. Benedetti et al. Plasma Phys. Control. Fusion 60.1: 014002 (2017)]) are computed and used in the laser envelope advance. Keeping the phase should be more accurate, but can cause numerical issues in the presence of strong depletion/frequency shift.

* ``lasers.solver_type`` (`string`) optional (default `multigrid`)
    Type of solver for the laser envelope solver, either ``fft`` or ``multigrid``.
    Currently, the approximation that the phase is evaluated on-axis only is made with both solvers.
    With the multigrid solver, we could drop this assumption.
    For now, the fft solver should be faster, more accurate and more stable, so only use the multigrid one with care.

* ``lasers.MG_tolerance_rel`` (`float`) optional (default `1e-4`)
    Relative error tolerance of the multigrid solver used for the laser pulse.

* ``lasers.MG_tolerance_abs`` (`float`) optional (default `0.`)
    Absolute error tolerance of the multigrid solver used for the laser pulse.

* ``lasers.MG_verbose`` (`int`) optional (default `0`)
    Level of verbosity of the multigrid solver used for the laser pulse.

* ``lasers.MG_average_rhs`` (`0` or `1`) optional (default `1`)
    Whether to use the most stable discretization for the envelope solver.

* ``lasers.3d_on_host`` (`0` or `1`) optional (default `0`)
    When running on GPU: whether the 3D array containing the laser envelope is stored in host memory (CPU, slower but large memory available) or in device memory (GPU, faster but less memory available).

* ``lasers.input_file`` (`string`) optional (default `""`)
    Path to an openPMD file containing a laser envelope.
    The file should comply with the `LaserEnvelope extension of the openPMD-standard <https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md>`__, as generated by `LASY <https://github.com/LASY-org/LASY>`__.
    Currently supported geometries: 3D or cylindrical profiles with azimuthal decomposition.
    The laser pulse is injected in the HiPACE++ simulation so that the beginning of the temporal profile from the file corresponds to the head of the simulation box, and time (in the file) is converted to space (HiPACE++ longitudinal coordinate) with ``z = -c*t + const``.
    If this parameter is set, then the file will be used to initialize all lasers instead of using a gaussian profile.

* ``lasers.openPMD_laser_name`` (`string`) optional (default `laserEnvelope`)
    Name of the laser envelope field inside the openPMD file to be read in.

* ``lasers.iteration`` (`int`) optional (default `0`)
    Iteration of the openPMD file to be read in.

* ``<laser name>.a0`` (`float`) optional (default `0`)
    Peak normalized vector potential of the laser pulse.

* ``<laser name>.position_mean`` (3 `float`) optional (default `0 0 0`)
    The mean position of the laser in `x, y, z`.

* ``<laser name>.w0`` (2 `float`) optional (default `0 0`)
    The laser waist in `x, y`.

* ``<laser name>.L0`` (`float`) optional (default `0`)
    The laser pulse length in `z`. Use either the pulse length or the pulse duration ``<laser name>.tau``.

* ``<laser name>.tau`` (`float`) optional (default `0`)
    The laser pulse duration. The pulse length will be set to `laser.tau`:math:`/c_0`.
    Use either the pulse length or the pulse duration.

* ``<laser name>.focal_distance`` (`float`)
    Distance at which the laser pulse if focused (in the z direction, counted from laser initial position).

Diagnostic parameters
---------------------

There are different types of diagnostics in HiPACE++. The standard diagnostics are compliant with the openPMD standard. The
in-situ diagnostics allow for fast analysis of large beams or the plasma particles.

* ``diagnostic.output_period`` (`integer`) optional (default `0`)
    Output period for standard beam and field diagnostics. Field or beam specific diagnostics can overwrite this parameter.
    No output is given for ``diagnostic.output_period = 0``.

Beam diagnostics
^^^^^^^^^^^^^^^^

* ``diagnostic.beam_output_period`` (`integer`) optional (default `0`)
    Output period for the beam. No output is given for ``diagnostic.beam_output_period = 0``.
    If ``diagnostic.output_period`` is defined, that value is used as the default for this.

* ``diagnostic.beam_data`` (`string`) optional (default `all`)
    Names of the beams written to file, separated by a space. The beam names need to be ``all``,
    ``none`` or a subset of ``beams.names``.

Field diagnostics
^^^^^^^^^^^^^^^^^

* ``diagnostic.names`` (`string`) optional (default `lev0`)
    The names of all field diagnostics, separated by a space.
    Multiple diagnostics can be used to limit the output to only a few relevant regions to save on file size.
    To run without field diagnostics, choose the name ``no_field_diag``.
    If mesh refinement is used, the default becomes ``lev0 lev1`` or ``lev0 lev1 lev2``.

* ``<diag name> or diagnostic.level`` (`integer`) optional (default `0`)
    From which mesh refinement level the diagnostics should be collected.
    If ``<diag name>`` is equal to ``lev1``, the default for this parameter becomes 1 etc.

* ``<diag name>.output_period`` (`integer`) optional (default `0`)
    Output period for fields. No output is given for ``<diag name>.output_period = 0``.
    If ``diagnostic.output_period`` is defined, that value is used as the default for this.

* ``<diag name> or diagnostic.diag_type`` (`string`)
    Type of field output. Available options are `xyz`, `xz`, `yz`. `xyz` generates a 3D field
    output. Use 3D output with parsimony, it may increase disk Space usage and simulation time
    significantly. `xz` and `yz` generate 2D field outputs at the center of the y-axis and
    x-axis, respectively. In case of an even number of grid points, the value will be averaged
    between the two inner grid points.

* ``<diag name> or diagnostic.coarsening`` (3 `int`) optional (default `1 1 1`)
    Coarsening ratio of field output in x, y and z direction respectively. The coarsened output is
    obtained through first order interpolation.

* ``<diag name> or diagnostic.include_ghost_cells`` (`bool`) optional (default `0`)
    Whether the field diagnostics should include ghost cells.

* ``<diag name> or diagnostic.field_data`` (`string`) optional (default `all`)
    Names of the fields written to file, separated by a space. The field names need to be ``all``,
    ``none`` or a subset of ``ExmBy EypBx Ez Bx By Bz Psi``. For the predictor-corrector solver,
    additionally ``jx jy jz rhomjz`` are available, which are the current and charge densities of the
    plasma and the beam, with ``rhomjz`` equal to :math:`\rho-j_z/c`.
    For the explicit solver, the current and charge densities of the beam and
    for all plasmas are separated: ``jx_beam jy_beam jz_beam`` and ``jx jy rhomjz`` are available.
    If ``rho`` is explicitly mentioned as ``field_data``, it will be deposited by the plasma
    to be available as a diagnostic.
    When a laser pulse is used, the real and imaginary parts of the laser complex envelope are written in ``laser_real`` and ``laser_imag``, respectively.
    The plasma proper density (n/gamma) is then also accessible via ``chi``.

* ``<diag name> or diagnostic.patch_lo`` (3 `float`) optional (default `-infinity -infinity -infinity`)
    Lower limit for the diagnostic grid.

* ``<diag name> or diagnostic.patch_hi`` (3 `float`) optional (default `infinity infinity infinity`)
    Upper limit for the diagnostic grid.

In-situ diagnostics
^^^^^^^^^^^^^^^^^^^

Besides the standard diagnostics, fast in-situ diagnostics are available. They are most useful when beams with large numbers of particles are used, as the important moments can be calculated in-situ (during the simulation) to largely reduce the simulation's analysis.
In-situ diagnostics compute slice quantities (1 number per quantity per longitudinal cell).
For particle beams, they can be used to calculate the main characterizing beam parameters (width, energy spread, emittance, etc.), from which most common beam parameters (e.g. slice and projected emittance, etc.) can be computed. Additionally, the plasma particle properties (e.g, the temperature) can be calculated.

For particle beams, the following quantities are calculated per slice and stored:
``sum(w), [x], [x^2], [y], [y^2], [z], [z^2], [ux], [ux^2], [uy], [uy^2], [uz], [uz^2], [x*ux], [y*uy], [z*uz], [ga], [ga^2], np``.
For plasma particles, the following quantities are calculated per slice and stored:
``sum(w), [x], [x^2], [y], [y^2], [ux], [ux^2], [uy], [uy^2], [uz], [uz^2], [ga], [ga^2], np``.
Thereby, "[]" stands for averaging over all particles in the current slice,
"w" stands for weight, "ux" is the normalized momentum in the x direction, "ga" is the Lorentz factor.
Averages and totals over all slices are also provided for convenience under the
respective ``average`` and ``total`` subcategories.

Additionally, some metadata is also available:
``time, step, n_slices, charge, mass, z_lo, z_hi, normalized_density_factor``.
``time`` and ``step`` refers to the physical time of the simulation and step number of the
current timestep.
``n_slices`` is the number of slices in the zeta direction.
``charge`` and ``mass`` relate to a single particle and are for example equal to the
electron charge and mass.
``z_lo`` and ``z_hi`` are the lower and upper bounds of the z-axis of the simulation domain
specified in the input file and can be used to generate a z/zeta-axis for plotting (note that they corresponds to mesh nodes, while the data is cell-centered).
``normalized_density_factor`` is equal to ``dx * dy * dz`` in normalized units and 1 in
SI units. It can be used to convert ``sum(w)``, which specifies the particle density in normalized
units and particle weight in SI units, to the particle weight in both unit systems.

The data is written to a file at ``<insitu_file_prefix>/reduced_<beam/plasma name>.<MPI rank number>.txt``.
The in-situ diagnostics file format consists of a header part in ASCII containing a JSON object.
When this is parsed into Python it can be converted to a NumPy structured datatype.
The rest of the file, following immediately after the closing ``}``, is in binary format and
contains all of the in-situ diagnostics along with some metadata. This part can be read using the
structured datatype of the first section.
Use ``hipace/tools/read_insitu_diagnostics.py`` to read the files using this format. Functions to calculate the most useful properties are also provided in that file.

* ``<beam name> or beams.insitu_period`` (`int`) optional (default ``0``)
    Period of the beam in-situ diagnostics. `0` means no beam in-situ diagnostics.

* ``<beam name> or beams.insitu_file_prefix`` (`string`) optional (default ``"diags/insitu"``)
    Path of the beam in-situ output. Must not be the same as `hipace.file_prefix`.

* ``<plasma name> or plasmas.insitu_period`` (`int`) optional (default ``0``)
    Period of the plasma in-situ diagnostics. `0` means no plasma in-situ diagnostics.

* ``<plasma name> or plasmas.insitu_file_prefix`` (`string`) optional (default ``"plasma_diags/insitu"``)
    Path of the plasma in-situ output. Must not be the same as `hipace.file_prefix`.
