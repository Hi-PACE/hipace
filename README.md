# hipace

![linux](https://github.com/Hi-PACE/hipace/workflows/linux/badge.svg?branch=development&event=push)
![macOS](https://github.com/Hi-PACE/hipace/workflows/macos/badge.svg?branch=development&event=push)

Highly efficient Plasma Accelerator Emulation, quasistatic particle-in-cell code


## Users

### Install

To do: users can install HiPACE with the following one-liners...

### Usage

Usage docs.


## Developers

### Dependencies

HiPACE depends on the following popular third party software.
Please see installation instructions below in the *Developers* section.

- a mature [C++14](https://en.wikipedia.org/wiki/C%2B%2B14) compiler: e.g. GCC 5, Clang 3.6 or newer
- [CMake 3.14.0+](https://cmake.org/)
- [AMReX *development*](https://amrex-codes.github.io): we automatically download and compile a copy of AMReX
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
spack add ccache
spack add cmake
spack add fftw
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
brew install ccache
brew install cmake
brew install fftw
brew install libomp
brew install pkg-config  # for fftw
brew install open-mpi
```

Now, `cmake --version` should be at version 3.14.0 or newer.

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
mkdir -p build
cd build

# find dependencies & configure
cmake ..

# build using up to four threads
make -j 4

# run tests
ctest --output-on-failure
```

You can inspect and modify build options after running `cmake ..` with either
```bash
ccmake .
```

or by providing arguments to the CMake call: `cmake .. -D<OPTION_A>=<VALUE_A> -D<OPTION_B>=<VALUE_B>`

| CMake Option                 | Default & Values                           | Description                                         |
|------------------------------|--------------------------------------------|-----------------------------------------------------|
| `CMAKE_BUILD_TYPE`           | **RelWithDebInfo**/Release/Debug           | Type of build, symbols & optimizations              |
| `HiPACE_COMPUTE`             | **NONE**/CUDA/OMP                          | Parallel, on-node computing backend                 |
| `HiPACE_PRECISION`           | **double**/single                          | Floating point precision (single/double)            |
| `HiPACE_amrex_repo`          | `https://github.com/AMReX-Codes/amrex.git` | Repository URI to pull and build AMReX from         |
| `HiPACE_amrex_branch`        | `development`                              | Repository branch for `HiPACE_amrex_repo`           |
| `HiPACE_amrex_internal`      | **ON**/OFF                                 | Needs a pre-installed AMReX library if set to `OFF` |

For example, one can also build against a local AMReX git repo.
Assuming AMReX' source is located in `$HOME/src/amrex` and changes are committed into a branch such as `my-amrex-branch` then pass to `cmake` the arguments `-DHiPACE_amrex_repo=file://$HOME/src/amrex -DHiPACE_amrex_branch=my-amrex-branch`.

HiPACE benefits from further standardized options in AMReX, which are [documented in detail in the AMReX manual](https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options).
A commonly needed option is:

| CMake Option                 | Default & Values                           | Description              |
|------------------------------|--------------------------------------------|--------------------------|
| `ENABLE_MPI`                 | **ON**/OFF                                 | Multi-node (MPI) support |


## Run a first simulation and look at the results

After compiling HiPACE (see above), from the HiPACE root directory, execute
```bash
cd examples
../build/bin/HiPACE inputs # Run the simulation with fields and plasma and beam particles
./show.py # Plot results with yt, and save in img.png
```
