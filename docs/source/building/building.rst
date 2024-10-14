.. _build-source:

.. these lines below don't seem to work anymore, fixing it by hand
.. raw:: html

   <style>
   .rst-content .section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Build/install HiPACE++
======================

.. _Developers:

Developers
----------

If you are new to CMake, `this short tutorial <https://hsf-training.github.io/hsf-training-cmake-webpage/>`__ from the HEP Software foundation is the perfect place to get started with it. If you just want to use CMake to build the project, jump into sections *1. Introduction*, *2. Building with CMake* and *9. Finding Packages*.

Dependencies
------------

HiPACE++ depends on the following popular third party software.
Please see installation instructions below in the Developers section.

- a mature `C++17 <https://en.wikipedia.org/wiki/C%2B%2B14>`__ compiler: e.g. GCC 7, Clang 7, NVCC 11.0, MSVC 19.15 or newer
- `CMake 3.24.0+ <https://cmake.org/>`__
- `AMReX development <https://amrex-codes.github.io>`__: we automatically download and compile a copy of AMReX
- `openPMD-api 0.16.0+ <https://github.com/openPMD/openPMD-api>`__: we automatically download and compile a copy of openPMD-api

  - `HDF5 <https://support.hdfgroup.org/HDF5>`__ 1.8.13+ (optional; for ``.h5`` file support)
  - `ADIOS2 <https://github.com/ornladios/ADIOS2>`__ 2.7.0+ (optional; for ``.bp`` file support)

Platform-dependent, at least one of the following:

- `CUDA Toolkit 11.0+ <https://developer.nvidia.com/cuda-downloads>`__: for NVIDIA GPU support (see `matching host-compilers <https://gist.github.com/ax3l/9489132>`__)
- `ROCm 5.2+ <https://github.com/RadeonOpenCompute/ROCm>`__: for AMD GPU support
- `FFTW3 <http://www.fftw.org/>`__: for CPUs (only used serially, but multi-threading supported; *not* needed for GPUs)

Optional dependencies include:

- `MPI 3.0+ <https://www.mpi-forum.org/docs/>`__: for multi-node and/or multi-GPU execution
- `OpenMP 3.1+ <https://www.openmp.org>`__: for threaded CPU execution
- `CCache <https://ccache.dev>`__: to speed up rebuilds (needs 3.7.9+ for CUDA)

Please choose **one** of the installation methods below to get started:

HPC platforms
-------------

If you want to use HiPACE++ on a specific high-performance computing (HPC) systems, jump directly to our :ref:`HPC system-specific documentation <install-hpc>`.

.. _install-spack:

.. only:: html

   .. image:: spack.svg
      :width: 32px
      :align: left


Using the Spack package manager
-------------------------------

The dependencies can be installed via the package manager
`Spack <https://spack.readthedocs.io/en/latest/>`__ (macOS/Linux):

.. code-block:: bash

   spack env create hipace-dev
   spack env activate hipace-dev
   spack add adios2  # for .bp file support
   spack add ccache
   spack add cmake
   spack add fftw
   spack add hdf5    # for .h5 file support
   spack add mpi
   spack add pkgconfig  # for fftw
   # optional:
   # spack add cuda
   spack install

(in new terminals, re-activate the environment with ``spack env activate hipace-dev`` again)

.. note::
   On Linux distributions, the InstallError ``"OpenMPI requires both C and Fortran compilers"`` can occur because the Fortran compilers are sometimes not set automatically in Spack.
   To fix this, the Fortran compilers must be set manually using ``spack config edit compilers`` (more information can be found `here <https://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration>`__).
   For GCC, the flags ``f77 : null`` and ``fc : null`` must be set to ``f77 : gfortran`` and ``fc : gfortran``.

   On macOS, a Fortran compiler like gfortran might be missing and must be installed by hand to fix this issue.

.. _install-brew:

.. only:: html

   .. image:: brew.svg
      :width: 32px
      :align: left

Using the Brew package manager
------------------------------

The dependencies can be installed via the package manager
`Homebrew <https://brew.sh/>`__ (macOS/Linux):


.. code-block:: bash

   brew update
   brew install adios2  # for .bp file support
   brew install ccache
   brew install cmake
   brew install fftw
   brew install hdf5-mpi  # for .h5 file support
   brew install libomp
   brew install pkg-config  # for fftw
   brew install open-mpi

Now, ``cmake --version`` should be at version 3.24.0 or newer.

Configure your compiler
-----------------------

For example, using a GCC on macOS:

.. code-block:: bash

   export CC=$(which gcc)
   export CXX=$(which g++)


If you also want to select a CUDA compiler:

.. code-block:: bash

   export CUDACXX=$(which nvcc)
   export CUDAHOSTCXX=$(which g++)


Build & Test
------------

If you have not downloaded HiPACE++ yet, please clone it from GitHub via

.. code-block:: bash

   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # or choose your preferred path

From the base of the HiPACE++ source directory, execute:

.. code-block:: bash

   # find dependencies & configure
   cmake -S . -B build

   # build using up to four threads
   cmake --build build -j 4

   # run tests
   (cd build; ctest --output-on-failure)

Note: the from_file tests require the openPMD-api with python bindings. See
`documentation of the openPMD-api <https://openpmd-api.readthedocs.io/>`__ for more information.
An executable HiPACE++ binary with the current compile-time options encoded in its file name will be created in ``bin/``.
Additionally, a `symbolic link <https://en.wikipedia.org/wiki/Symbolic_link>`__ named ``hipace`` can be found in that directory, which points to the last built HiPACE++ executable. You can inspect and modify build options after running `cmake ..` with either

.. code-block:: bash

   ccmake build

or by providing arguments to the CMake call

.. code-block:: bash

   cmake -S . -B build -D<OPTION_A>=<VALUE_A> -D<OPTION_B>=<VALUE_B>


=============================  ========================================  =========================================================
 CMake Option                  Default & Values                          Description
-----------------------------  ----------------------------------------  ---------------------------------------------------------
 ``CMAKE_BUILD_TYPE``          RelWithDebInfo/**Release**/Debug          Type of build, symbols & optimizations
 ``HiPACE_COMPUTE``            NOACC/CUDA/SYCL/HIP/**OMP**               On-node, accelerated computing backend
 ``HiPACE_MPI``                **ON**/OFF                                Multi-node support (message-passing)
 ``HiPACE_PRECISION``          SINGLE/**DOUBLE**                         Floating point precision (single/double)
 ``HiPACE_OPENPMD``            **ON**/OFF                                openPMD I/O (HDF5, ADIOS2)
 ``HiPACE_PUSHER``             **LEAPFROG**/AB5                          Use leapfrog or fifth-order Adams-Bashforth plasma pusher
=============================  ========================================  =========================================================

HiPACE++ can be configured in further detail with options from AMReX, which are documented in the `AMReX manual <https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options>`__.

**Developers** might be interested in additional options that control dependencies of HiPACE++.
By default, the most important dependencies of HiPACE++ are automatically downloaded for convenience:

===========================  ==================================================  =============================================================
CMake Option                 Default & Values                                    Description
---------------------------  --------------------------------------------------  -------------------------------------------------------------
``HiPACE_amrex_src``         *None*                                              Path to AMReX source directory (preferred if set)
``HiPACE_amrex_repo``        ``https://github.com/AMReX-Codes/amrex.git``        Repository URI to pull and build AMReX from
``HiPACE_amrex_branch``      ``development``                                     Repository branch for ``HiPACE_amrex_repo``
``HiPACE_amrex_internal``    **ON**/OFF                                          Needs a pre-installed AMReX library if set to ``OFF``
``HiPACE_openpmd_mpi``       ON/OFF (default is set to value of ``HiPACE_MPI``)  Build openPMD with MPI support, although I/O is always serial
``HiPACE_openpmd_src``       *None*                                              Path to openPMD-api source directory (preferred if set)
``HiPACE_openpmd_repo``      ``https://github.com/openPMD/openPMD-api.git``      Repository URI to pull and build openPMD-api from
``HiPACE_openpmd_branch``    ``0.16.0``                                          Repository branch for ``HiPACE_openpmd_repo``
``HiPACE_openpmd_internal``  **ON**/OFF                                          Needs a pre-installed openPMD-api library if set to ``OFF``
``AMReX_LINEAR_SOLVERS``     ON/**OFF**                                          Compile AMReX multigrid solver.
===========================  ==================================================  =============================================================

For example, one can also build against a local AMReX copy.
Assuming AMReX' source is located in ``$HOME/src/amrex``, add the ``cmake`` argument ``-DHiPACE_amrex_src=$HOME/src/amrex``.
Relative paths are also supported, e.g. ``-DHiPACE_amrex_src=../amrex``.

Or build against an AMReX feature branch of a colleague.
Assuming your colleague pushed AMReX to ``https://github.com/WeiqunZhang/amrex/`` in a branch ``new-feature`` then pass to ``cmake`` the arguments: ``-DHiPACE_amrex_repo=https://github.com/WeiqunZhang/amrex.git -DHiPACE_amrex_branch=new-feature``.

You can speed up the install further if you pre-install these dependencies, e.g. with a package manager.
Set ``-DHiPACE_<dependency-name>_internal=OFF`` and add installation prefix of the dependency to the environment variable `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html>`__.
Please see the short CMake tutorial that we linked in the :ref:`Developers` section if this sounds new to you.

Documentation
-------------

The documentation is written at the `RST <https://sphinx-tutorial.readthedocs.io/step-1/>`__ format, to compile the documentation locally use

.. code-block:: bash

   cd docs
   # optional:                                 --user
   python3 -m pip install -r requirements.txt          # only the first time
   make html
   open build/html/index.html

The last line would work on MacOS. On another platform, open the html file with your favorite browser.
