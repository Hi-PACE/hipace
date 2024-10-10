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

============ ========================= =====================
**variable** **name**                  **Value**
q_e          elementary charge         1.602176634e-19
m_e          electron mass             9.1093837015e-31
m_p          proton mass               1.67262192369e-27
epsilon0     vacuum permittivity       8.8541878128e-12
mu0          vacuum permeability       1.25663706212e-06
clight       speed of light            299'792'458.
hbar         reduced Planck constant   1.054571817e-34
r_e          classical electron radius 2.817940326204929e-15
============ ========================= =====================

For a list of supported functions see the
`AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parser>`__.
Sometimes it is necessary to use double-quotes around expressions, especially when providing them
as command line parameters. Multi-line expressions are allowed if surrounded by double-quotes.
For input parameters of type ``string``, ``my_constants`` can be putinside curly braces ``{...}`` to directly paste them into the input parameter.
If what is inside the braces is not a ``my_constants`` it will be evaluated as an expression using the parser.

General parameters
------------------

* ``hipace.normalized_units`` (`bool`) optional (default `0`)
    Whether the simulation uses the normalized unit system commonly used in wakefield acceleration, see e.g. `chapter 2 of this reference <https://iopscience.iop.org/article/10.1088/0741-3335/56/8/084012>`__. Otherwise, the code assumes SI (Système International) unit system.

* ``random_seed`` (`integer`) optional (default `1`)
    Passes a seed to the AMReX random number generator. This allows for reproducibility of random events such as randomly generated beams, ionization, and collisions.
    Note that on GPU, since the order of operations is not ensured, providing a seed does not guarantee reproducibility to the level of machine precision.

* ``use_previous_rng`` (`bool`) optional (default `0`)
    If set to `1`, the seed of the random number generator is computed as was done previous to `Pull Request 1081 <https://github.com/Hi-PACE/hipace/pull/1081>`__.
    In particular, this seed depends on the number of ranks used for the simulation.
    If ``random_seed`` is specified, it takes precedence and ``use_previous_rng`` is not used.
    This option is off by default and should only be used for backward compatibility.

* ``hipace.verbose`` (`int`) optional (default `0`)
    Level of verbosity.

      * ``hipace.verbose = 1``, prints only the time steps, which are computed.

      * ``hipace.verbose = 2`` additionally prints the number of iterations in the
        predictor-corrector loop, as well as the B-Field error at each slice.

      * ``hipace.verbose = 3`` also prints the number of particles that violate the quasi-static
        approximation and were neglected at each slice. It prints the number of ionized particles,
        if ionization occurred. It also adds additional information if beams
        are read in from file.

* ``hipace.do_device_synchronize`` (`int`) optional (default `0`)
    Level of synchronization on GPU.

      * ``hipace.do_device_synchronize = 0``, synchronization happens only when necessary.

      * ``hipace.do_device_synchronize = 1``, synchronizes most functions (all that are profiled
        via ``HIPACE_PROFILE``)

* ``amrex.the_arena_is_managed`` (`bool`) optional (default `0`)
    Whether managed memory is used. Note that large simulations sometimes only fit on a GPU if managed memory is used,
    but generally it is recommended to not use it.

* ``amrex.omp_threads``  (``system``, ``nosmt`` or positive integer; default is ``nosmt``)
    An integer number can be set in lieu of the ``OMP_NUM_THREADS`` environment variable to control the number of OpenMP threads to use for the ``OMP`` compute backend on CPUs.
    By default, we use the ``nosmt`` option, which overwrites the OpenMP default of spawning one thread per logical CPU core, and instead only spawns a number of threads equal to the number of physical CPU cores on the machine.
    If set, the environment variable ``OMP_NUM_THREADS`` takes precedence over ``system`` and ``nosmt``, but not over integer numbers set in this option.

* ``comms_buffer.on_gpu`` (`bool`) optional (default `0`)
    Whether the buffers that hold the beam and the 3D laser envelope should be allocated on the GPU (device memory).
    By default they will be allocated on the CPU (pinned memory).
    Setting this option to `1` is necessary to take advantage of GPU-Enabled MPI, however for this
    additional enviroment variables need to be set depending on the system.

* ``comms_buffer.async_memcpy`` (`bool`) optional (default `1`)
    When using a GPU and setting ``comms_buffer.on_gpu = 0``, this option will allow the data
    transfer between the CPU and GPU for communications to be asynchronous instead of blocking.
    This can improve performance in a typical situation where the CPU-GPU link has relatively
    low bandwidth at the cost of some GPU memory and a reduced maximum number of MPI ranks.

* ``comms_buffer.max_size_GiB`` (`float`) optional (default `inf`)
    How many Gibibytes of beam particles and laser slices can be stored in the communications buffer
    on each rank. This setting offers an alternative to ``comms_buffer.max_leading_slices``
    and ``comms_buffer.max_trailing_slices``. Note that the amount specified here may be slightly
    exeeded in practice. If there are more time steps than ranks, this parameter must be chosen
    such that between all ranks there is enough capacity to store every beam particle and
    laser slice to avoid a deadlock, i.e.
    ``comms_buffer.max_size_GiB * nranks > beam_size + laser_size``.

* ``comms_buffer.max_leading_slices`` (`int`) optional (default `inf`)
    How many slices of beam particles can be received and stored in advance.

* ``comms_buffer.max_trailing_slices`` (`int`) optional (default `inf`)
    How many slices of beam particles can be stored before being sent. Using
    ``comms_buffer.max_leading_slices`` and ``comms_buffer.max_trailing_slices`` will in principle
    limit the amount of asynchronousness in the parallel communication and may thus reduce performance.
    However it may be necessary to set these parameters to avoid all slices accumulating on a single
    rank that would run out of memory (out of CPU or GPU memory depending on ``comms_buffer.on_gpu``).
    If there are more time steps than ranks, these parameters must be chosen such that between all
    ranks there is enough capacity to store every slice to avoid a deadlock, i.e.
    ``comms_buffer.max_trailing_slices * nranks > nslices``.

* ``comms_buffer.pre_register_memory`` (`bool`) optional (default `false`)
    On some platforms, such as JUWELS booster, the memory passed into MPI needs to be
    registered to the network card, which can take a long time. When using this option, all ranks
    can do this at once in initialization instead of one after another
    as part of the communication pipeline.

* ``hipace.do_shared_depos`` (`bool`) optional (default `false`)
    Whether to use shared memory current deposition on GPU.

* ``hipace.do_tiling`` (`bool`) optional (default `true`)
    Whether to use tiling, when running on CPU.
    Currently, this option only affects plasma operations (gather, push and deposition).
    The tile size can be set with ``hipace.tile_size``.

* ``hipace.tile_size`` (`int`) optional (default `32`)
    Tile size for beam and plasma current deposition, when running on CPU
    and tiling is activated (``hipace.do_tiling = 1``).

* ``hipace.depos_order_xy`` (`int`) optional (default `2`)
    Transverse particle shape order. Currently, `0,1,2,3` are implemented.

* ``hipace.depos_order_z`` (`int`) optional (default `0`)
    Longitudinal particle shape order. Currently, only `0` is implemented.

* ``hipace.depos_derivative_type`` (`int`) optional (default `2`)
    Type of derivative used in explicit deposition. `0`: analytic, `1`: nodal, `2`: centered

* ``hipace.do_beam_jx_jy_deposition`` (`bool`) optional (default `1`)
    Using the default, the beam deposits all currents ``Jx``, ``Jy``, ``Jz``. Using
    ``hipace.do_beam_jx_jy_deposition = 0`` disables the transverse current deposition of the beams.

* ``hipace.do_beam_jz_minus_rho`` (`bool`) optional (default `0`)
    Whether the beam contribution to :math:`j_z-c\rho` is calculated and used when solving for Psi (used to caculate the transverse fields Ex-By and Ey+Bx).
    if 0, this term is assumed to be 0 (a good approximation for an ultra-relativistic beam in the z direction with small transverse momentum).

* ``hipace.interpolate_neutralizing_background`` (`bool`) optional (default `0`)
    Whether the neutralizing background from plasmas should be interpolated from level 0
    to higher MR levels instead of depositing it on all levels.

* ``hipace.output_input`` (`bool`) optional (default `0`)
    Print all input parameters before running the simulation.
    If a parameter is present multiple times then the last occurrence will be used.
    Note that this will include some default AMReX parameters.

Geometry
--------

* ``amr.n_cell`` (3 `integer`)
    Number of cells in x, y and z.
    With the explicit solver (default), the number of cells in the x and y directions must be either :math:`2^n-1` (common values are 511, 1023, 2047, best configuration for performance) or :math:`2^n` where :math:`n` is an integer. Some other values might work, like :math:`3 \times 2^n-1`, but use at your own risk.

* ``amr.max_level`` (`integer`) optional (default `0`)
    Maximum level of mesh refinement. Currently, mesh refinement is supported up to level
    `2`. Note, that the mesh refinement algorithm is still in active development and should be used with care.

* ``geometry.prob_lo`` (3 `float`)
    Lower end of the simulation box in x, y and z.

* ``geometry.prob_hi`` (3 `float`)
    Higher end of the simulation box in x, y and z.

* ``boundary.field`` (`string`)
    Type of boundary condition used to fill the ghost cells of the fields.
    Possible values:

        * ``Dirichlet`` The field value in ghost cells stays zero.

        * ``Periodic`` The field value in ghost cells is filled using periodic continuation of the domain.
            This option should usually be selected only in combination with periodic field solvers.

        * ``Open`` Uses a Taylor approximation of the Greens function to solve the Poisson equations with
            open boundary conditions. Only available with the predictor-corrector solver.

* ``boundary.particle`` (`string`)
    Type of boundary condition used for particles.
    Possible values:

        * ``Reflecting`` Particles are reflected into the domain where they exited.

        * ``Periodic`` Particles enter the domain on the opposite side where they exit.

        * ``Absorbing`` Particles exiting the domain are deleted.

* ``boundary.particle_lo`` (2 `float`) optional (default `<first two values of geometry.prob_lo>`)
    The lower location of the domain boundary the particles experience. By default, this is equal
    to the boundary of the fields however it may be shrunk to reduce noise originating from
    the boundary, especially when using open boundary conditions.

* ``boundary.particle_hi`` (2 `float`) optional (default `<first two values of geometry.prob_hi>`)
    The upper location of the domain boundary the particles experience.
    See ``boundary.particle_lo``.

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

* ``lasers.n_cell`` (2 `integer`)
    Number of cells in x and y for the laser grid.
    The number of cells in the zeta direction is calculated from ``patch_lo`` and ``patch_hi``.

* ``lasers.patch_lo`` (3 `float`)
    Lower end of the laser grid in x, y and z.

* ``lasers.patch_hi`` (3 `float`)
    Upper end of the laser grid in x, y and z.

Time step
---------

* ``max_step`` (`integer`) optional (default `0`)
    Maximum number of time steps. `0` means that the 0th time step will be calculated, which are the
    fields of the initial beams.

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

* ``fields.poisson_solver`` (`string`) optional (default CPU: `FFTDirichletDirect`, GPU: `FFTDirichletFast`)
    Which Poisson solver to use for ``Psi``, ``Ez`` and ``Bz``. The ``predictor-corrector`` BxBy
    solver also uses this poisson solver for ``Bx`` and ``By`` internally. Available solvers are:

      * ``FFTDirichletDirect`` Use the discrete sine transformation that is directly implemented
        by FFTW to solve the Poisson equation with Dirichlet boundary conditions.
        This option is only available when compiling for CPUs with FFTW.
        Preferred resolution: :math:`2^N-1`.

      * ``FFTDirichletExpanded`` Perform the discrete sine transformation by symmetrically
        expanding the field to twice its size.
        Preferred resolution: :math:`2^N-1`.

      * ``FFTDirichletFast`` Perform the discrete sine transformation using a fast sine transform
        algorithm that uses FFTs of the same size as the fields.
        Preferred resolution: :math:`2^N-1`.

      * ``MGDirichlet`` Use the HiPACE++ multigrid solver to solve the Poisson equation with
        Dirichlet boundary conditions.
        Preferred resolution: :math:`2^N` and :math:`2^N-1`.

      * ``FFTPeriodic`` Use FFTs to solve the Poisson equation with Periodic boundary conditions.
        Note that this does not work with features that change the boundary values,
        like mesh refinement or open boundaries.
        Preferred resolution: :math:`2^N`.

* ``fields.do_symmetrize`` (`bool`) optional (default `0`)
    Symmetrizes current and charge densities transversely before the field solve.
    Each cell at (`x`, `y`) is averaged with cells at (`-x`, `y`), (`x`, `-y`) and (`-x`, `-y`).

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

* ``<plasma name> or plasmas.ppc`` (2 `integer`)
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
    If set, the plasma charge gets multiplied by this number. If the plasma species is not ionizable,
    the initial ionization level is set to 1.

* ``<plasma name>.ionization_product`` (`string`) optional (default "")
    Name of the plasma species that contains the new electrons that are produced
    when this plasma gets ionized. Only needed if this plasma is ionizable.

* ``<plasma name> or plasmas.neutralize_background`` (`bool`) optional (default `1`)
    Whether to add a neutralizing background of immobile particles of opposite charge.

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

* ``<plasma name> or plasmas.n_subcycles`` (`int`) optional (default `1`)
    Number of sub-cycles within the plasma pusher. Currently only implemented for the leapfrog pusher. Must be larger or equal to 1. Sub-cycling is needed if plasma particles move
    significantly in the transverse direction during a single longitudinal cell. If they move too many cells such that they do not sample certain small transverse structures in the wakefields, sub-cycling is needed and fixes the issue.

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

* ``<plasma name> or plasmas.fine_patch(x,y)`` (`int`) optional (default `0`)
    When using mesh refinement it can be helpful to increase the number of particles per cell drastically
    in a small part of the domain. For this parameter a function of ``x`` and ``y`` needs to be specified
    that evaluates to ``1`` where the number of particles per cell should be higher and ``0`` everywhere else.
    For example use ``plasmas.fine_patch(x,y) = "sqrt(x^2+y^2) < 10"`` to specify a circle around ``x=0, y=0``
    with a radius of ``10``. Note that the function is evaluated at the cell centers of the level zero grid.

* ``<plasma name> or plasmas.fine_ppc`` (2 `int`) optional (default `0 0`)
    The number of plasma particles per cell in x and y inside the fine plasma patch. This must be
    divisible by the ppc outside the fine patch in both directions.

* ``<plasma name> or plasmas.fine_transition_cells`` (`int`) optional (default `5`)
    Number of cells that are used just outside of the fine plasma patch to smoothly transition
    between the low and high ppc regions. More transition cells produce less noise but
    require more particles.

* ``<plasma name> or plasmas.prevent_centered_particle`` (`bool`) optional (default `0`)
    When ``amr.n_cell`` and the plasma ppc are both odd, a plasma particle is initialized
    in the exact center of the domain. A symmetric beam also initialized at the center of the domain will not be
    able to push this particle away, causing the plasma particle to pass through the beam and
    increasing its emittance. Enabling this setting causes all plasma particles to be
    initialized half a cell to the side so that no plasma particle will be at the exact center of
    the domain. However, this will also result in a gap at the domain boundary,
    which can lead to noise.

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
    The injection type for the particle beam. Currently available are ``fixed_weight_pdf``, ``fixed_weight``, ``fixed_ppc``,
    and ``from_file``.
    ``fixed_weight_pdf`` generates a beam with a fixed number of particles with a constant weight where
    the transverse profile is Gaussian and the longitudinal profile is arbitrary according to a
    user-specified probability density function. It is more general and faster, and uses
    less memory than ``fixed_weight``.
    ``fixed_weight`` generates a Gaussian beam with a fixed number of particles with a constant weight.
    ``fixed_ppc`` generates a beam with a fixed number of particles per cell and
    varying weights. It can be either a Gaussian or a flattop beam.
    ``from_file`` reads a beam from openPMD files.

* ``<beam name>.element`` (`string`) optional (default `electron`)
    The Physical Element of the plasma. Sets charge, mass and, if available,
    the specific Ionization Energy of each state.
    Currently available options are: ``electron``, ``positron``, and ``proton``.

* ``<beam name>.mass`` (`float`) optional (default `m_e`)
    The mass of beam particles. Can also be set with ``<beam name>.element``. Must be `>0`.

* ``<beam name>.charge`` (`float`) optional (default `-q_e`)
    The charge of a beam particle. Can also be set with ``<beam name>.element``.

* ``<beam name>.n_subcycles`` (`int`) optional (default `10`)
    Number of sub-cycles performed in the beam particle pusher. The particles will be pushed
    ``n_subcycles`` times with a time step of `dt/n_subcycles`. This can be used to improve accuracy
    in highly non-linear focusing fields.

* ``<beam name> or beams.external_E(x,y,z,t)`` (3 `float`) optional (default `0. 0. 0.`)
    External electric field applied to beam particles as functions of x, y, z and t.
    The components represent Ex, Ey and Ez respectively.
    Note that z refers to the location of the beam particle inside the moving frame of reference
    (zeta) and t to the physical time of the current timestep.

* ``<beam name> or beams.external_B(x,y,z,t)`` (3 `float`) optional (default `0. 0. 0.`)
    External magnetic field applied to beam particles as functions of x, y, z and t.
    The components represent Bx, By and Bz respectively.
    Note that z refers to the location of the beam particle inside the moving frame of reference
    (zeta) and t to the physical time of the current timestep.

* ``<beam name>.do_z_push`` (`bool`) optional (default `1`)
    Whether the beam particles are pushed along the z-axis. The momentum is still fully updated.
    Note: using ``do_z_push = 0`` results in unphysical behavior.

* ``<beam name> or beams.reorder_period`` (`int`) optional (default `0`)
    Reorder particles periodically to speed-up current deposition and particle push on GPU.
    A good starting point is a period of 1 to reorder beam particles on every timestep.
    To disable reordering set this to 0. For very narrow beams the sorting may take longer than
    the time saved in the beam push and deposition.

* ``<beam name> or beams.reorder_idx_type`` (2 `int`) optional (default `0 0` or `1 1`)
    Change if beam particles are binned to cells (0), nodes (1) or both (2)
    for both x and y direction as part of the reordering.
    The ideal index type is different for beam push and beam deposition so some experimentation
    may be required to find the overall fastest setting for a specific simulation.

Option: ``fixed_weight_pdf``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``<beam name>.num_particles`` (`int`)
    Number of constant weight particles to generate the beam.

* ``<beam name>.pdf`` (`float`)
    Longitudinal density profile of the beam, given as a probability density function
    (the transverse profile is Gaussian). This is a parser function of z, giving the charge density
    integrated in both transverse directions `x` and `y` (this is proportional to the beam current
    profile in the limit :math:`v_z \simeq c`). The probability density function is automatically
    normalized, and combined with ``<beam name>.total_charge`` or ``<beam name>.density`` within
    the code to generate the absolute beam profile.
    Examples (assuming ``z_center``, ``z_std``, ``z_length``, ``z_slope``, ``z_min`` and ``z_max``
    are defined with ``my_constants``):
    - Gaussian: ``exp(-0.5*((z-z_center)/z_std)^2)``
    - Cosine: ``(cos(2*pi*(z-z_center)/z_length)+1)*(2*abs(z-z_center)<z_length)``
    - Trapezoidal: ``(z<z_max)*(z>z_min)*(1+z_slope*z)``

* ``<beam name>.total_charge`` (`float`)
    Total charge of the beam (either ``total_charge`` or ``density`` must be specified).
    Only available when running in SI units.
    The absolute value of this parameter is used when initializing the beam.
    Note that in contrast to the ``fixed_weight`` injection type, using ``<beam name>.radius`` or
    a special pdf to emulate ``z_min`` and ``z_max`` will result in beam particles being redistributed to
    other locations rather than being deleted. Therefore, the resulting beam will have exactly the
    specified total charge, but cutting a significant fraction of the charge is not recommended.

* ``<beam name>.density`` (`float`)
    Peak density of the beam (either ``total_charge`` or ``density`` must be specified).
    The absolute value of this parameter is used when initializing the beam.
    Note that this is the peak density of the analytical profile specified by `pdf`, `position_mean` and
    `position_std`, within the limits of the resolution of the numerical evaluation of the pdf. The actual
    resulting beam profile consists of randomly distributed particles and will likely feature density
    fluctuations exceeding the specified peak density.

* ``<beam name>.position_mean`` (2 `float`)
    The mean position of the beam in ``x, y``, separated by a space. Both values can be a function of z.
    To generate a tilted beam use
    ``<beam name>.position_mean = "x_center+(z-z_center)*dx_per_dzeta" "y_center+(z-z_center)*dy_per_dzeta"``.

* ``<beam name>.position_std`` (2 `float`)
    The rms size of the of the beam in ``x, y``, separated by a space. Both values can be a function of z.

* ``<beam name>.u_mean`` (3 `float`)
    The mean normalized momentum of the beam in ``x, y, z``, separated by a space. All values can be a function of z.
    Normalized momentum is equal to :math:`= \gamma \beta = \frac{p}{m c}`. An electron beam with a momentum of 1 GeV/c
    has a u_mean of ``0 0 1956.951198`` while a proton beam with the same momentum has a u_mean of ``0 0 1.065788933``.

* ``<beam name>.u_std`` (3 `float`)
    The rms normalized momentum of the beam in ``x, y, z``, separated by a space. All values can be a function of z.

* ``<beam name>.do_symmetrize`` (`bool`) optional (default `0`)
    Symmetrizes the beam in the transverse phase space. For each particle with (`x`, `y`, `ux`,
    `uy`), three further particles are generated with (`-x`, `y`, `-ux`, `uy`), (`x`, `-y`, `ux`,
    `-uy`), and (`-x`, `-y`, `-ux`, `-uy`). The total number of particles will still be
    ``beam_name.num_particles``, therefore this option requires that the beam particle number must be
    divisible by 4.

* ``<beam name>.z_foc`` (`float`) optional (default `0.`)
    Distance at which the beam will be focused, calculated from the position at which the beam is initialized.
    The beam is assumed to propagate ballistically in-between.

* ``<beam name>.radius`` (`float`) optional (default `infinity`)
    Maximum radius ``<beam name>.radius`` :math:`= \sqrt{x^2 + y^2}` within that particles are
    injected. If ``<beam name>.density`` is specified, beam particles outside of the radius get
    deleted. If ``<beam name>.total_charge`` is specified, beam particles outside of the radius get
    new random transverse positions to conserve the total charge.

* ``<beam name>.pdf_ref_ratio`` (`int`) optional (default `4`)
    Into how many segments the pdf is divided per zeta slice for its first-order numerical evaluation.

Option: ``fixed_weight``
^^^^^^^^^^^^^^^^^^^^^^^^

* ``<beam name>.num_particles`` (`int`)
    Number of constant weight particles to generate the beam.

* ``<beam name>.profile`` (`string`) optional (default `gaussian`)
    Beam profile.
    Possible options are ``can`` (uniform longitudinally, Gaussian transversally)
    and ``gaussian`` (Gaussian in all directions).

* ``<beam name>.total_charge`` (`float`)
    Total charge of the beam. Note: Either ``total_charge`` or ``density`` must be specified.
    The absolute value of this parameter is used when initializing the beam.
    Note that ``<beam name>.zmin``, ``<beam name>.zmax`` and ``<beam name>.radius`` can reduce the total charge.

* ``<beam name>.density`` (`float`)
    Peak density of the beam. Note: Either ``total_charge`` or ``density`` must be specified.
    The absolute value of this parameter is used when initializing the beam.

* ``<beam name>.position_mean`` (3 `float`)
    The mean position of the beam in ``x, y, z``, separated by a space.
    The x and y directions can be functions of ``z``. To generate a tilted beam use
    ``<beam name>.position_mean = "x_center+(z-z_ center)*dx_per_dzeta" "y_center+(z-z_ center)*dy_per_dzeta" "z_center"``.

* ``<beam name>.position_std`` (3 `float`)
    The rms size of the of the beam in ``x, y, z``, separated by a space.

* ``<beam name>.u_mean`` (3 `float`)
    The mean normalized momentum of the beam in ``x, y, z``, separated by a space.
    Normalized momentum is equal to :math:`= \gamma \beta = \frac{p}{m c}`. An electron beam with a momentum of 1 GeV/c
    has a u_mean of ``0 0 1956.951198`` while a proton beam with the same momentum has a u_mean of ``0 0 1.065788933``.

* ``<beam name>.u_std`` (3 `float`)
    The rms normalized momentum of the beam in ``x, y, z``, separated by a space.

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

* ``<beam name>.z_foc`` (`float`) optional (default `0.`)
    Distance at which the beam will be focused, calculated from the position at which the beam is initialized.
    The beam is assumed to propagate ballistically in-between.

* ``<beam name>.zmin`` (`float`) (default `-infinity`)
    Minimum in `z` at which particles are injected.

* ``<beam name>.zmax`` (`float`) (default `infinity`)
    Maximum in `z` at which particles are injected.

* ``<beam name>.radius`` (`float`) (default `infinity`)
    Maximum radius ``<beam name>.radius`` :math:`= \sqrt{x^2 + y^2}` within that particles are
    injected.

* ``<beam name> or beams.initialize_on_cpu`` (`bool`) optional (default `0`)
    Whether to initialize the beam on the CPU instead of the GPU.
    Initializing the beam on the CPU can be much slower but is necessary if the full beam does not fit into GPU memory.

Option: ``fixed_ppc``
^^^^^^^^^^^^^^^^^^^^^

* ``<beam name>.ppc`` (3 `int`) (default `1 1 1`)
    Number of particles per cell in `x`-, `y`-, and `z`-direction to generate the beam.

* ``<beam name>.profile`` (`string`)
    Beam profile.
    Possible options are ``flattop`` (flat-top radially and longitudinally),
    ``gaussian`` (Gaussian in all directions),
    or ``parsed`` (arbitrary analytic function provided by the user).
    When ``parsed``, ``<beam name>.density(x,y,z)`` must be specified.

* ``<beam name>.density`` (`float`)
    Peak density of the beam.
    The absolute value of this parameter is used when initializing the beam.

* ``<beam name>.density(x,y,z)`` (`float`)
    The density profile of the beam, as a function of spatial dimensions `x`, `y` and `z`.
    This function uses the parser, see above.

* ``<beam name>.min_density`` (`float`) optional (default `0`)
    Particles with a density less than or equal to the minimal density won't be injected.
    The absolute value of this parameter is used when initializing the beam.

* ``<beam name>.position_mean`` (3 `float`)
    The mean position of the beam in ``x, y, z``, separated by a space.

* ``<beam name>.position_std`` (3 `float`)
    The rms size of the of the beam in ``x, y, z``, separated by a space.

* ``<beam name>.u_mean`` (3 `float`)
    The mean normalized momentum of the beam in ``x, y, z``, separated by a space.
    Normalized momentum is equal to :math:`= \gamma \beta = \frac{p}{m c}`. An electron beam with a momentum of 1 GeV/c
    has a u_mean of ``0 0 1956.951198`` while a proton beam with the same momentum has a u_mean of ``0 0 1.065788933``.

* ``<beam name>.u_std`` (3 `float`)
    The rms normalized momentum of the beam in ``x, y, z``, separated by a space.

* ``<beam name>.random_ppc`` (3 `bool`) optional (default `0 0 0`)
    Whether the position in `(x y z)` of the particles is randomized within the cell.

* ``<beam name>.zmin`` (`float`) (default `-infinity`)
    Minimum in `z` at which particles are injected.

* ``<beam name>.zmax`` (`float`) (default `infinity`)
    Maximum in `z` at which particles are injected.

* ``<beam name>.radius`` (`float`) (default `infinity`)
    Maximum radius ``<beam name>.radius`` :math:`= \sqrt{x^2 + y^2}` within that particles are
    injected.

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

* ``<beam name> or beams.initialize_on_cpu`` (`bool`) optional (default `0`)
    Whether to initialize the beam on the CPU instead of the GPU.
    Initializing the beam on the CPU can be much slower but is necessary if the full beam does not fit into GPU memory.

SALAME algorithm
^^^^^^^^^^^^^^^^

HiPACE++ features the Slicing Advanced Loading Algorithm for Minimizing Energy Spread (SALAME) to generate a beam profile that
automatically loads the wake optimally, i.e., so that the initial wakefield is flattened by the charge of the beam. Important note:
In the algorithm, the weight of the beam particles is adjusted while the plasma response is computed. Since the beam is written to file
**before** the plasma response is calculated, the SALAME beam has incorrect weights in the 0th time step.
For more information on the algorithm, see the corresponding publication `S. Diederichs et al., Phys. Rev. Accel. Beams 23, 121301 (2020) <https://doi.org/10.1103/PhysRevAccelBeams.23.121301>`__

* ``<beam name>.do_salame`` (`bool`) optional (default `0`)
    If turned on, the per-slice beam weight in the first time-step is adjusted such that the Ez field is uniform in the beam.
    This ignores the contributions to jx, jy and rho from the beam in the first time-step.
    It is recommended to use this option with a fixed weight can beam.
    If a gaussian beam profile is used, then the zmin and zmax parameters should be used.

* ``hipace.salame_n_iter`` (`int`) optional (default `5`)
    The maximum number of iterations the SALAME algorithm should do when it is used.

* ``hipace.salame_relative_tolerance`` (`float`) optional (default `1e-4`)
    Relative error tolerance to finish SALAME iterations early.

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

Laser parameters
----------------

The laser profile is defined by :math:`a(x,y,z) = a_0 * \mathrm{exp}[-(x^2/w0_x^2 + y^2/w0_y^2 + z^2/L0^2)]`.
The model implemented is the one from [C. Benedetti et al. Plasma Phys. Control. Fusion 60.1: 014002 (2017)].
Unlike for ``beams`` and ``plasmas``, all the laser pulses are currently stored on the same array,
which you can find in the output openPMD file as a complex array named `laserEnvelope`.
Parameters starting with ``lasers.`` apply to all laser pulses, parameters starting with ``<laser name>`` apply to a single laser pulse.

* ``lasers.names`` (list of `string`) optional (default `no_laser`)
    The names of the laser pulses, separated by a space.
    To run without a laser, choose the name ``no_laser``.

* ``lasers.use_phase`` (`bool`) optional (default `true`)
    Whether the phase terms (:math:`\theta` in Eq. (6) of [C. Benedetti et al. Plasma Phys. Control. Fusion 60.1: 014002 (2017)]) are computed and used in the laser envelope advance. Keeping the phase should be more accurate, but can cause numerical issues in the presence of strong depletion/frequency shift.

* ``lasers.interp_order`` (`int`) optional (default `1`)
    Transverse shape order for the laser to field interpolation of aabs and
    the field to laser interpolation of chi. Currently, `0,1,2,3` are implemented.

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

* ``<laser name>.init_type`` (list of `string`) optional (default `gaussian`)
    The initialisation method of laser. Possible options are:

      Option: ``gaussian`` (default) the laser is initialised with an ideal gaussian pulse.

      * ``<laser name>.a0`` (`float`) optional (default `0`)
          Peak normalized vector potential of the laser pulse.

      * ``lasers.lambda0`` (`float`)
          Wavelength of the laser pulses. Currently, all pulses must have the same wavelength.

      * ``<laser name>.position_mean`` (3 `float`) optional (default `0 0 0`)
          The mean position of the laser in `x, y, z`.

      * ``<laser name>.w0`` (2 `float`) optional (default `0 0`)
          The laser waist in `x, y`.

      * ``<laser name>.L0`` (`float`) optional (default `0`)
          The laser pulse length in `z`. Use either the pulse length or the pulse duration ``<laser name>.tau``.

      * ``<laser name>.tau`` (`float`) optional (default `0`)
          The laser pulse duration. The pulse length is set to `laser.tau`:math:`*c_0`.
          Use either the pulse length or the pulse duration.

      * ``<laser name>.focal_distance`` (`float`)
          Distance at which the laser pulse is focused (in the z direction, counted from laser initial position).

      * ``<laser name>.propagation_angle_yz`` (`float`) optional (default `0`)
          Propagation angle of the pulse in the yz plane (0 is along the z axis)

      Option: ``from_file`` the laser is loaded from an openPMD file.

      * ``<laser name>.input_file`` (`string`) optional (default `""`)
          Path to an openPMD file containing a laser envelope.
          The file should comply with the `LaserEnvelope extension of the openPMD-standard <https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md>`__, as generated by `LASY <https://github.com/LASY-org/LASY>`__.
          Currently supported geometries: 3D or cylindrical profiles with azimuthal decomposition.
          The laser pulse is injected in the HiPACE++ simulation so that the beginning of the temporal profile from the file corresponds to the head of the simulation box, and time (in the file) is converted to space (HiPACE++ longitudinal coordinate) with ``z = -c*t + const``.
          If this parameter is set, then the file is used to initialize all lasers instead of using a gaussian profile.

      * ``<laser name>.openPMD_laser_name`` (`string`) optional (default `laserEnvelope`)
          Name of the laser envelope field inside the openPMD file to be read in.

      * ``<laser name>.iteration`` (`int`) optional (default `0`)
          Iteration of the openPMD file to be read in.

      Option: ``parser``, the laser is initialized with the expression of the complex envelope function.

      * ``<laser name>.laser_real(x,y,z)`` optional (`string`) (default `""`)
          Expression for the real part of the laser envelope in `x, y, z`.

      * ``<laser name>.laser_imag(x,y,z)`` optional (`string`) (default `""`)
          Expression for the imaginary part of the laser envelope `x, y, z`.

      * ``lasers.lambda0`` (`float`)
          Wavelength of the laser pulses. Currently, all pulses must have the same wavelength.

Diagnostic parameters
---------------------

There are different types of diagnostics in HiPACE++. The standard diagnostics are compliant with the openPMD standard. The
in-situ diagnostics allow for fast analysis of large beams or the plasma particles.

* ``diagnostic.output_period`` (`integer`) optional (default `0`)
    Output period for standard beam and field diagnostics. Field or beam specific diagnostics can overwrite this parameter.
    No output is given for ``diagnostic.output_period = 0``.

* ``hipace.file_prefix`` (`string`) optional (default `diags/hdf5/`)
    Path of the output.

* ``hipace.openpmd_backend`` (`string`) optional (default `h5`)
    OpenPMD backend. This can either be ``h5``, ``bp``, or ``json``. The default is chosen by what is
    available. If both Adios2 and HDF5 are available, ``h5`` is used. Note that ``json`` is extremely
    slow and is not recommended for production runs.

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
    Depending on whether mesh refinement or a laser is used, the default becomes
    a subset of ``lev0 lev1 lev2 laser_diag``.

* ``<diag name> or diagnostic.base_geometry`` (`string`) optional (default `level_0`)
    Which geometry the diagnostics should be based on.
    Available geometries are `level_0`, `level_1`, `level_2` and `laser`,
    depending on if MR or a laser is used.
    If ``<diag name>`` is equal to ``lev0 lev1 lev2 laser_diag``, the default for this parameter
    becomes ``level_0 level_1 level_2 laser``respectively.

* ``<diag name>.output_period`` (`integer`) optional (default `0`)
    Output period for fields. No output is given for ``<diag name>.output_period = 0``.
    If ``diagnostic.output_period`` is defined, that value is used as the default for this.

* ``<diag name> or diagnostic.diag_type`` (`string`)
    Type of field output. Available options are `xyz`, `xz`, `yz` and `xy_integrated`.
    `xyz` generates a 3D field output.
    Use 3D output with parsimony, it may increase disk Space usage and simulation time significantly.
    `xz` and `yz` generate 2D field outputs at the center of the y-axis and
    x-axis, respectively. In case of an even number of grid points, the value is averaged
    between the two inner grid points.
    `xy_integrated` generates 2D field output that has been integrated along the `z` axis, i.e.,
    it is the sum of the 2D field output over all slices multiplied with `dz`.

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
    If ``rho`` is explicitly mentioned as ``field_data``, it is deposited by the plasma
    to be available as a diagnostic. Similarly if ``rho_<plasma name>`` is explicitly mentioned,
    the charge density of that plasma species will be separately available as a diagnostic.
    When a laser pulse is used, the laser complex envelope ``laserEnvelope`` is available
    in the ``laser`` base geometry.
    The plasma proper density (n/gamma) is then also accessible via ``chi``.
    A field can be removed from the list, for example, after it has been included through ``all``,
    by adding ``remove_<field name>`` after it has been added. If a field is added and removed
    multiple times, the last occurrence takes precedence.

* ``<diag name> or diagnostic.patch_lo`` (3 `float`) optional (default `-infinity -infinity -infinity`)
    Lower limit for the diagnostic grid.

* ``<diag name> or diagnostic.patch_hi`` (3 `float`) optional (default `infinity infinity infinity`)
    Upper limit for the diagnostic grid.

* ``hipace.deposit_rho`` (`bool`) optional (default `0`)
    If the charge density ``rho`` of the plasma should be deposited so that it is available as a diagnostic.
    Otherwise only ``rhomjz`` equal to :math:`\rho-j_z/c` will be available.
    If ``rho`` is explicitly mentioned in ``diagnostic.field_data``, then the default will become `1`.

* ``hipace.deposit_rho_individual`` (`bool`) optional (default `0`)
    This option works similar to ``hipace.deposit_rho``,
    however the charge density from every plasma species will be deposited into individual fields
    that are accessible as ``rho_<plasma name>`` in ``diagnostic.field_data``.

In-situ diagnostics
^^^^^^^^^^^^^^^^^^^

Besides the standard diagnostics, fast in-situ diagnostics are available. They are most useful when beams with large numbers of particles are used, as the important moments can be calculated in-situ (during the simulation) to largely reduce the simulation's analysis.
In-situ diagnostics compute slice quantities (1 number per quantity per longitudinal cell).
For particle beams, they can be used to calculate the main characterizing beam parameters (width, energy spread, emittance, etc.), from which most common beam parameters (e.g. slice and projected emittance, etc.) can be computed. Additionally, the plasma particle properties (e.g, the temperature) can be calculated.
For particle quantities, "[...]" stands for averaging over all particles in the current slice;
for grid quantities, "[...]" stands for integrating over all cells in the current slice.

For particle beams, the following quantities are calculated per slice and stored:
``sum(w), [x], [x^2], [y], [y^2], [z], [z^2], [ux], [ux^2], [uy], [uy^2], [uz], [uz^2], [x*ux], [y*uy], [z*uz], [x*uy], [y*ux], [ux/uz], [uy/uz], [ga], [ga^2], np``.
For plasma particles, the following quantities are calculated per slice and stored:
``sum(w), [x], [x^2], [y], [y^2], [ux], [ux^2], [uy], [uy^2], [uz], [uz^2], [ga], [ga^2], np``.
Thereby, "w" stands for weight, "ux" is the normalized momentum in the x direction, "ga" is the Lorentz factor.
Averages and totals over all slices are also provided for convenience under the
respective ``average`` and ``total`` subcategories.

For the field in-situ diagnostics, the following quantities are calculated per slice and stored:
``[Ex^2], [Ey^2], [Ez^2], [Bx^2], [By^2], [Bz^2], [ExmBy^2], [EypBx^2], [jz_beam], [Ez*jz_beam]``.
These quantities can be used to calculate the energy stored in the fields.

For the laser in-situ diagnostics, the following quantities are calculated per slice and stored:
``max(|a|^2), [|a|^2], [|a|^2*x], [|a|^2*x*x], [|a|^2*y], [|a|^2*y*y], axis(a)``.
Thereby, ``max(|a|^2)`` is the highest value of ``|a|^2`` in the current slice
and ``axis(a)`` gives the complex value of the laser envelope, in the center of every slice.

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

* ``<beam name> or beams.insitu_radius`` (`float`) optional (default ``infinity``)
    Maximum radius ``<beam name>.insitu_radius`` :math:`= \sqrt{x^2 + y^2}` within which particles are
    used for the calculation of the insitu diagnostics.

* ``<plasma name> or plasmas.insitu_period`` (`int`) optional (default ``0``)
    Period of the plasma in-situ diagnostics. `0` means no plasma in-situ diagnostics.

* ``<plasma name> or plasmas.insitu_file_prefix`` (`string`) optional (default ``"plasma_diags/insitu"``)
    Path of the plasma in-situ output. Must not be the same as `hipace.file_prefix`.

* ``<plasma name> or plasmas.insitu_radius`` (`float`) optional (default ``infinity``)
    Maximum radius ``<plasma name>.insitu_radius`` :math:`= \sqrt{x^2 + y^2}` within which particles are
    used for the calculation of the insitu diagnostics.

* ``fields.insitu_period`` (`int`) optional (default ``0``)
    Period of the field in-situ diagnostics. `0` means no field in-situ diagnostics.

* ``fields.insitu_file_prefix`` (`string`) optional (default ``"diags/field_insitu"``)
    Path of the field in-situ output. Must not be the same as `hipace.file_prefix`.

* ``lasers.insitu_period`` (`int`) optional (default ``0``)
    Period of the laser in-situ diagnostics. `0` means no laser in-situ diagnostics.

* ``lasers.insitu_file_prefix`` (`string`) optional (default ``"diags/laser_insitu"``)
    Path of the laser in-situ output. Must not be the same as `hipace.file_prefix`.

Additional physics
------------------

Additional physics describe the physics modules implemented in HiPACE++ that go beyond the standard electromagnetic equations.
This includes ionization (see plasma parameters), binary collisions, and radiation reactions. Since all of these require the actual plasma density,
they need a background density in SI units, if the simulation runs in normalized units.

* ``hipace.background_density_SI`` (`float`) optional
    Background plasma density in SI units. Certain physical modules (collisions, ionization, radiation reactions) depend on the actual background density.
    Hence, in normalized units, they can only be included, if a background plasma density in SI units is provided using this input parameter.

Binary collisions
^^^^^^^^^^^^^^^^^

WARNING: this module is in development.

HiPACE++ proposes an implementation of [Perez et al., Phys. Plasmas 19, 083104 (2012)], inherited from WarpX,
for collisions between plasma-plasma and beam-plasma.
As collisions depend on the physical density, in normalized units `hipace.background_density_SI` must be specified.

* ``hipace.collisions`` (list of `strings`) optional
    List of names of binary Coulomb collisions.
    Each will represent collisions between 2 species.

* ``<collision name>.species`` (two `strings`) optional
    The name of the two species for which collisions should be included.
    This can either be plasma-plasma or beam-plasma collisions. For plasma-plasma collisions, the species can be the same to model collisions within a species.
    The names must be in `plasmas.names` or `beams.names` (for beam-plasma collisions).

* ``<collision name>.CoulombLog`` (`float`) optional (default `-1.`)
    Coulomb logarithm used for this collision.
    If not specified, the Coulomb logarithm is determined from the temperature in each cell.

Radiation reaction
^^^^^^^^^^^^^^^^^^

Whether the energy loss due to classical radiation reaction of beam particles is calculated.

* ``<beam name> or beams.do_radiation_reaction`` (`bool`) optional (default `0`)
    Whether the beam particles undergo energy loss due to classical radiation reaction.
    The implemented radiation reaction model is based on this publication: `M. Tamburini et al., NJP 12, 123005 <https://doi.org/10.1088/1367-2630/12/12/123005>`__
    In normalized units, `hipace.background_density_SI` must be specified.

Spin tracking
-------------

Track the spin of each beam particle as it is rotated by the electromagnetic fields using the
Thomas-Bargmann-Michel-Telegdi (TBMT) model, see
[Z. Gong et al., Matter and Radiation at Extremes 8.6 (2023), https://doi.org/10.1063/5.0152382]
for the details of the implementation.
This will add three extra components to each beam particle to store the spin and output
those as part of the beam diagnostic as ``spin/x, spin/y, spin/z``
or beam in-situ diagnostic as ``[sx], [sx^2], [sy], [sy^2], [sz], [sz^2]``.

* ``<beam name> or beams.do_spin_tracking`` (`bool`) optional (default `0`)
    Enable spin tracking

* ``<beam name> or beams.initial_spin`` (3 `float`)
    Initial spin ``sx sy sz`` of all particles. The length of the three components is normalized to one.

* ``<beam name> or beams.spin_anom`` (`bool`) optional (default `0.00115965218128`)
    The anomalous magnetic moment. The default value is the moment for electrons.
