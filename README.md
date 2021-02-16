# hipace

![linux](https://github.com/Hi-PACE/hipace/workflows/linux/badge.svg?branch=development&event=push)
<!-- ![macOS](https://github.com/Hi-PACE/hipace/workflows/macos/badge.svg?branch=development&event=push) -->

Highly efficient Plasma Accelerator Emulation, quasistatic particle-in-cell code

Note: this page is outdated, in order to get the latest documentation please do

```bash
   cd doc
   pip install -r requirements.txt # only the first time
   make html
   open build/html/index.html # or any way to open this HTML file.
```

## Users

### Install

To do: users can install HiPACE with the following one-liners...
(This needs to be done once we go open source, so we can set up conda-forge, Spack et al.)

### Usage

To do: Usage docs.


## Developers

If you are new to CMake, [this short tutorial](https://hsf-training.github.io/hsf-training-cmake-webpage/) from the HEP Software foundation is the perfect place to get started with it.

If you just want to use CMake to build the project, jump into sections *1. Introduction*, *2. Building with CMake* and *9. Finding Packages*.

### Dependencies

HiPACE depends on the following popular third party software.
Please see installation instructions below in the *Developers* section.

- a mature [C++14](https://en.wikipedia.org/wiki/C%2B%2B14) compiler: e.g. GCC 5, Clang 3.6 or newer
- [CMake 3.15.0+](https://cmake.org/)
- [AMReX *development*](https://amrex-codes.github.io): we automatically download and compile a copy of AMReX
- [openPMD-api *dev*](https://github.com/openPMD/openPMD-api): we automatically download and compile a copy of openPMD-api
  - [HDF5](https://support.hdfgroup.org/HDF5) 1.8.13+ (optional; for `.h5` file support)
  - [ADIOS2](https://github.com/ornladios/ADIOS2) 2.6.0+ (optional; for `.bp` file support)
- Nvidia GPU support:
  - [CUDA Toolkit 9.0+](https://developer.nvidia.com/cuda-downloads) (see [matching host-compilers](https://gist.github.com/ax3l/9489132))
- CPU-only:
  - [FFTW3](http://www.fftw.org/) (only used serially; *not* needed for Nvidia GPUs)

Optional dependencies include:
- [MPI 3.0+](https://www.mpi-forum.org/docs/): for multi-node and/or multi-GPU execution
- [OpenMP 3.1+](https://www.openmp.org): for threaded CPU execution (currently not fully accelerated)
- [CCache](https://ccache.dev): to speed up rebuilds (needs 3.7.9+ for CUDA)

### Install Dependencies

macOS/Linux:
```bash
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
```
(in new terminals, re-activate the environment with `spack env activate hipace-dev` again)

or macOS/Linux:
```bash
brew update
brew install adios2  # for .bp file support
brew install ccache
brew install cmake
brew install fftw
brew install hdf5-mpi  # for .h5 file support
brew install libomp
brew install pkg-config  # for fftw
brew install open-mpi
```

Now, `cmake --version` should be at version 3.15.0 or newer.

### Configure your compiler

For example, using a GCC on macOS:
```bash
export CC=$(which gcc)
export CXX=$(which g++)
```

If you also want to select a CUDA compiler:
```bash
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which g++)
```

### Build & Test

From the base of the HiPACE source directory, execute:
```bash
# find dependencies & configure
cmake -S . -B build

# build using up to four threads
cmake --build build -j 4

# run tests
(cd build; ctest --output-on-failure)
```

An executable HiPACE binary with the current compile-time options encoded in its file name will be created in ``bin/``.
Additionally, a `symbolic link <https://en.wikipedia.org/wiki/Symbolic_link>`_ named ``hipace`` can be found in that directory, which points to the last built HiPACE executable.

You can inspect and modify build options after first running `cmake` with either
```bash
ccmake build
```

or by providing arguments to the CMake call: `cmake -S . -B build -D<OPTION_A>=<VALUE_A> -D<OPTION_B>=<VALUE_B>`

| CMake Option                 | Default & Values                           | Description                                         |
|------------------------------|--------------------------------------------|-----------------------------------------------------|
| `CMAKE_BUILD_TYPE`           | **RelWithDebInfo**/Release/Debug           | Type of build, symbols & optimizations              |
| `HiPACE_COMPUTE`             | **NOACC**/CUDA/SYCL/HIP/OMP                | On-node, accelerated computing backend              |
| `HiPACE_MPI`                 | **ON**/OFF                                 | Multi-node support (message-passing)                |
| `HiPACE_PRECISION`           | SINGLE/**DOUBLE**                          | Floating point precision (single/double)            |
| `HiPACE_OPENPMD`             |  **ON**/OFF                                | openPMD I/O (HDF5, ADIOS2)                          |

HiPACE can be configured in further detail with options from AMReX, which are [documented in the AMReX manual](https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options).

**Developers** might be interested in additional options that control dependencies of HiPACE.
By default, the most important dependencies of HiPACE are automatically downloaded for convenience:

| CMake Option              | Default & Values                             | Description                                               |
|---------------------------|----------------------------------------------|-----------------------------------------------------------|
| `HiPACE_amrex_src`        | *None*                                       | Path to AMReX source directory (preferred if set)         |
| `HiPACE_amrex_repo`       | `https://github.com/AMReX-Codes/amrex.git`   | Repository URI to pull and build AMReX from               |
| `HiPACE_amrex_branch`     | `development`                                | Repository branch for `HiPACE_amrex_repo`                 |
| `HiPACE_amrex_internal`   | **ON**/OFF                                   | Needs a pre-installed AMReX library if set to `OFF`       |
| `HiPACE_openpmd_src`      | *None*                                       | Path to openPMD-api source directory (preferred if set)   |
| `HiPACE_openpmd_repo`     | `https://github.com/openPMD/openPMD-api.git` | Repository URI to pull and build openPMD-api from         |
| `HiPACE_openpmd_branch`   | `0.13.2`                                     | Repository branch for `HiPACE_openpmd_repo`               |
| `HiPACE_openpmd_internal` | **ON**/OFF                                   | Needs a pre-installed openPMD-api library if set to `OFF` |

For example, one can also build against a local AMReX copy.
Assuming AMReX' source is located in `$HOME/src/amrex`, add the `cmake` argument `-DHiPACE_amrex_src=$HOME/src/amrex`.
Relative paths are also supported, e.g. `-DHiPACE_amrex_src=../amrex`.

Or build against an AMReX feature branch of a colleague.
Assuming your colleague pushed AMReX to `https://github.com/WeiqunZhang/amrex/` in a branch `new-feature` then pass to `cmake` the arguments: `-DHiPACE_amrex_repo=https://github.com/WeiqunZhang/amrex.git -DHiPACE_amrex_branch=new-feature`.

You can speed up the install further if you pre-install these dependencies, e.g. with a package manager.
Set `-DHiPACE_<dependency-name>_internal=OFF` and add installation prefix of the dependency to the environment variable [CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html).
Please see the [short CMake tutorial that we linked above](#Developers) if this sounds new to you.


## Run a first simulation and look at the results

After compiling HiPACE (see above), from the HiPACE root directory, execute
```bash
cd examples/linear_wake/
../../build/bin/hipace inputs # Run the simulation with fields and plasma and beam particles
./analysis.py # Plot results with yt, and save in img.png
```

## Documentation

Hipace has a full (all functions and classes and their members, albeit sometimes basic) Doxygen-readable documentation. You can compile it with
```bash
cd docs
doxygen
open doxyhtml/index.html
```
The last line would work on MacOS. On another platform, open the html file with your favorite browser.

## Style and code conventions

- All new element (class, member of a class, struct, function) declared in a .H file must have a Doxygen-readable documentation
- Indent four spaces
- No tabs allowed
- No end-of-line whitespaces allowed
- Classes use CamelCase
- Objects use snake_case
- Lines should not have >100 characters
- The declaration and definition of a function should have a space between the function name and the first bracket (`my_function (...)`), function calls should not (`my_function(...)`).
  This is a convention introduce in AMReX so `git grep "my_function ("` returns only the declaration and definition, not the many function calls.
