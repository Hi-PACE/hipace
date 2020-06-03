# hipace

![linux](https://github.com/Hi-PACE/hipace/workflows/linux/badge.svg?branch=master&event=push)
![macOS](https://github.com/Hi-PACE/hipace/workflows/macos/badge.svg?branch=master&event=push)

Highly efficient Plasma Accelerator Emulation, quasistatic particle-in-cell code

## Developers

### Install Dependencies

macOS/Linux:
```bash
brew update
brew install ccache
brew install cmake
brew install libomp
brew install open-mpi
```

or macOS/Linux:
```bash
spack env create hipace-dev
spack env activate hipace-dev
spack add ccache
spack add cmake
spack add mpi
spack install
```
(in new terminals, re-activate the environment with `spack env activate hipace-dev` again)

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

cmake .. -DCMAKE_BUILD_TYPE=Debug

| CMake Option                 | Default & Values                           | Description                                         |
|------------------------------|--------------------------------------------|-----------------------------------------------------|
| `CMAKE_BUILD_TYPE`           | `RelWithDebInfo`/Release/Debug             | Type of build, symbols & optimizations              |
| `HiPACE_amrex_repo`          | `https://github.com/AMReX-Codes/amrex.git` | Repository URI to pull and build AMReX from         |
| `HiPACE_amrex_branch`        | `development`                              | Repository branch for `HiPACE_amrex_repo`           |
| `HiPACE_amrex_internal`      | **ON**/OFF                                 | Needs a pre-installed AMReX library if set to `OFF` |

For example, one can also build against a local AMReX git repo.
Assuming AMReX' source is located in `$HOME/src/amrex` and changes are committed into a branch such as `my-amrex-branch` then pass to `cmake` the arguments `-DHiPACE_amrex_repo=file://$HOME/src/amrex -DHiPACE_amrex_branch=my-amrex-branch`.

HiPACE benefits from further standardized options in AMReX, which are [documented in detail in the AMReX manual](https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options).
Commonly used options are:

| CMake Option                 | Default & Values                           | Description              |
|------------------------------|--------------------------------------------|--------------------------|
| `ENABLE_MPI`                 | **ON**/OFF                                 | Multi-node (MPI) support |
| `ENABLE_OMP`                 | ON/**OFF**                                 | OpenMP threads support   |
| `ENABLE_CUDA`                | ON/**OFF**                                 | Nvidia GPU support       |


### Run a simulation and look at the results

After compiling HiPACE (see above), from the HiPACE root directory, execute
```bash
cd examples
../build/bin/HiPACE inputs # Run the simulation with fields and plasma and beam particles
./show.py # Plot results with yt, and save in img.png
```
